import torch
import os
import numpy as np
import glob
from PIL import Image
import rasterio
from cvtorchvision import cvtransforms
import random
from torch.utils.data import Dataset, Subset, random_split
import torchvision
from einops import rearrange
import re

## function: split the dataset into train/val/test set
def dataset_split(dataset, val_pct, test_pct, seed=42):
    """Split a torch Dataset into train/val/test sets.
    If ``test_pct`` is not set then only train and validation splits are returned.
    Args:
        dataset: dataset to be split into train/val or train/val/test subsets
        val_pct: percentage of samples to be in validation set
        test_pct: (Optional) percentage of samples to be in test set
    Returns:
        a list of the subset datasets. Either [train, val] or [train, val, test]
    """
    if test_pct is None:
        val_length = int(len(dataset) * val_pct)
        train_length = len(dataset) - val_length
        return random_split(dataset, [train_length, val_length],generator=torch.Generator().manual_seed(seed))
    else:
        val_length = int(len(dataset) * val_pct)
        test_length = int(len(dataset) * test_pct)
        train_length = len(dataset) - (val_length + test_length)
        return random_split(dataset, [train_length, val_length, test_length],generator=torch.Generator().manual_seed(seed))

## class: subset dataset 
class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, transforms=None, transforms_color=None):
        self.dataset = dataset
        self.transforms = transforms
        self.transforms_color = transforms_color

    def __getitem__(self, index):

        image, mask, idx = self.dataset[index]

        if self.transforms_color:
            image_rgb = self.transforms_color(image[0:3])
            image_rgb = rearrange(image_rgb, "() c h w -> c h w")
            if image.shape[0]==3:
                image = image_rgb
            else:
                image = torch.cat((image_rgb,image[3:]))

        if self.transforms is not None:       
            image,mask = self.transforms(image,mask.float())
            mask = mask.to(torch.long)
            image = rearrange(image, "() c h w -> c h w")
            mask = rearrange(mask, "() () h w -> h w") 

        return image, mask, idx
    
    def __len__(self):
        return len(self.dataset)

## class: the main dataset
class IAAI_dataset(torch.utils.data.Dataset):

    def __init__(self,root,transforms=None,transforms_color=None,subset='all',split='train'):

        self.root = root # data dir
        self.transforms = transforms # data transforms
        self.subset = subset # use 'all' data or 'rgb' data
        self.split = split # if 'test' no masks available
        self.transforms_color = transforms_color # color transform for orthophoto

        ortho_filenames = os.listdir(os.path.join(self.root,'Orthophoto'))
        self.ids = []
        for ortho_file in ortho_filenames:
            idx = re.findall(r'\d+', ortho_file)
            self.ids.append(idx[0])
        self.length = len(self.ids)

    def __len__(self):
        return self.length

    def __getitem__(self,index):

        ''' read data '''
        ## image
        aspect_path = os.path.join(self.root,'Aspect','aspect_{}.png'.format(self.ids[index]))
        img_aspect = Image.open(aspect_path)

        dtm_path = os.path.join(self.root,'DTM','dtm_{}.png'.format(self.ids[index]))
        img_dtm = Image.open(dtm_path)

        flowacc_path = os.path.join(self.root,'Flow_Accum','flowacc_{}.png'.format(self.ids[index]))
        img_flowacc = Image.open(flowacc_path)

        flowdir_path = os.path.join(self.root,'Flow_Direction','flowdir_{}.png'.format(self.ids[index]))
        img_flowdir = Image.open(flowdir_path)                        

        ortho_path = os.path.join(self.root,'Orthophoto','ortho_{}.png'.format(self.ids[index]))
        img_ortho = Image.open(ortho_path)

        pcurv_path = os.path.join(self.root,'Prof_curv','pcurv_{}.png'.format(self.ids[index]))
        img_pcurv = Image.open(pcurv_path)

        slope_path = os.path.join(self.root,'Slope','slope_{}.png'.format(self.ids[index]))
        img_slope = Image.open(slope_path)

        tcurv_path = os.path.join(self.root,'Tang_curv','tcurv_{}.png'.format(self.ids[index]))
        img_tcurv = Image.open(tcurv_path)

        twi_path = os.path.join(self.root,'Topo_Wetness','twi_{}.png'.format(self.ids[index]))
        img_twi = Image.open(twi_path)

        if self.subset=='rgb':
            image = np.asarray(img_ortho).astype('float32') / 255.0
            
        elif self.subset=='all':
            image_rgb = np.asarray(img_ortho)
            image_others = np.dstack([
                np.asarray(img_aspect),
                np.asarray(img_dtm),
                np.asarray(img_flowacc),
                np.asarray(img_flowdir),
                np.asarray(img_pcurv),
                np.asarray(img_slope),
                np.asarray(img_tcurv),
                np.asarray(img_twi),                
            ])

            image = np.concatenate([image_rgb,image_others],-1).astype('float32') / 255.0

        image = torch.from_numpy(image.transpose(2,0,1))

        ## target
        if self.split == 'train':
            mask_dir = os.path.join(self.root,'Ground_truth')
            mask_path = os.path.join(mask_dir,'mask_{}.png'.format(self.ids[index]))
            mask = np.asarray(Image.open(mask_path)).copy()

            mask[mask==128] = 1
            mask[mask==255] = 2

            mask = torch.from_numpy(mask).long()
        else:
            mask = None

        ''' augment and transform data '''
        if self.transforms_color:
            image_rgb = self.transforms_color(image[0:3])
            image_rgb = rearrange(image_rgb, "() c h w -> c h w")
            
            if image.shape[0]==3:
                image = image_rgb
            else:
                image = torch.cat((image_rgb,image[3:]))

        if self.transforms is not None:                
            image,mask = self.transforms(image,mask.float())
            mask = mask.to(torch.long)
            image = rearrange(image, "() c h w -> c h w")
            mask = rearrange(mask, "() () h w -> h w") 

        if self.split=='test':
            return image, self.ids[index]
        else:
            return image, mask, self.ids[index]


    

if __name__=="__main__":
    ''' to check if the dataset class is correct '''
    
    import kornia.augmentation as K
    
    transforms_both = K.AugmentationSequential(
        K.RandomResizedCrop((512, 512), scale=(0.5, 1.0), p=1.0, keepdim=False),
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        data_keys=["input", "mask"],
    )

    train_dataset = IAAI_dataset(root='../data/Hackathon_data',
                                 subset='all',
                                 transforms=transforms_both
                                 )


    train_dataset, val_dataset, _ = dataset_split(train_dataset, val_pct=0.2, test_pct=0.0)

    dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=1,shuffle=False,num_workers=0,sampler=None,pin_memory=False, drop_last=False)
    print(len(train_dataset))
    for i,(image,mask,idx) in enumerate(dataloader):
        print(image.shape,image.dtype,mask.shape,mask.dtype, len(idx))