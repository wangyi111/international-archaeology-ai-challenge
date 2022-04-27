'''
Merge and inference with two models.
2022.04.27

'''

import os
import torch
from datasets.IAAI_dataset import IAAI_dataset,dataset_split,Subset
import segmentation_models_pytorch as smp
from tqdm import tqdm
import pdb
import argparse
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser('IAAI_inference', add_help=False)
parser.add_argument('--root_dir',type=str)
parser.add_argument('--out_dir',type=str)
parser.add_argument('--backbone', type=str, default='resnet50')
parser.add_argument('--resume_unet', default='', type=str, help='path to trained checkpoint')
parser.add_argument('--resume_deeplab', default='', type=str, help='path to trained checkpoint')
parser.add_argument('--subset', type=str, default='all')

args = parser.parse_args()

args.root_dir = 'data/Hackathon_200test'
args.out_dir = 'results/Preds_04/'
args.backbone = 'efficientnet-b5'
args.resume_unet = 'checkpoints/all_unet_efn_dice2_aug_attn_final/checkpoint_0099.pth.tar'
args.resume_deeplab = 'checkpoints/final_deeplab/checkpoint_0079.pth.tar'


''' prepare data '''
dataset = IAAI_dataset(args.root_dir,subset=args.subset,transforms=None,split='test')
#train_data, val_data, _ = dataset_split(dataset, val_pct=0.2, test_pct=0.0, seed=42)
dataloader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0,sampler=None,pin_memory=False, drop_last=False)

print('test_len: %d' % (len(dataloader)))

''' prepare model '''
if args.subset=='all':
    in_channels = 11
elif args.subset=='rgb':
    in_channels = 3


unet_model = smp.Unet(
        encoder_name=args.backbone,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=None,           # use `imagenet` pre-trained weights for encoder initialization
        in_channels=in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=3,                      # model output channels (number of classes in your dataset)
        decoder_attention_type='scse',
    )
deeplab_model = smp.DeepLabV3Plus(
        encoder_name=args.backbone,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=None,           # use `imagenet` pre-trained weights for encoder initialization
        in_channels=in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=3,                      # model output channels (number of classes in your dataset)
    )

''' load model weights '''
try:
    checkpoint1 = torch.load(args.resume_unet)
    unet_model.load_state_dict(checkpoint1['model_state_dict'])

    checkpoint2 = torch.load(args.resume_deeplab)
    deeplab_model.load_state_dict(checkpoint2['model_state_dict'])    

except:
    print('checkpoint load error!')
    raise NotImplementedError

unet_model.cuda()
deeplab_model.cuda()

''' inference with merged models '''
os.makedirs(args.out_dir,exist_ok=True)

print('Start testing...')
unet_model.eval()
deeplab_model.eval()

with torch.no_grad():
    for i, data in enumerate(tqdm(dataloader)):
        inputs,idx = data[0].cuda(),data[1]
        output1 = unet_model(inputs)
        output2 = deeplab_model(inputs)

        output1 = torch.nn.functional.softmax(output1,dim=1)
        output2 = torch.nn.functional.softmax(output2,dim=1)

        output = (output1 + output2) / 2.0

        prediction = torch.squeeze(torch.argmax(output,1)).detach().cpu().numpy()

        prediction[prediction==1] = 128
        prediction[prediction==2] = 255

        cv2.imwrite(os.path.join(args.out_dir, 'pred_{}.png'.format(idx[0])),prediction)