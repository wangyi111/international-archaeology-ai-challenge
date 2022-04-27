'''
Main script to train the models.

2022.04.27
'''

import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pdb
import argparse
import builtins
import math
import numpy as np
import cv2
import random
import sklearn.metrics as skm
from datasets.IAAI_dataset import IAAI_dataset,dataset_split,Subset
from torch.utils.tensorboard import SummaryWriter
import segmentation_models_pytorch as smp
import torchmetrics
import kornia.augmentation as K
from kornia.constants import Resample

class DiceLoss(nn.Module):
    def __init__(
        self, mode: str = "multiclass", ignore_index: int = 0, normalized: bool = True
    ):
        super().__init__()
        
        self.dice_loss = smp.losses.DiceLoss(mode=mode, ignore_index=ignore_index)

    def forward(self, preds, targets):
        return self.dice_loss(preds, targets)

class FocalLoss(nn.Module):
    def __init__(
        self, mode: str = "multiclass", ignore_index: int = 0, normalized: bool = False
    ):
        super().__init__()
        self.focal_loss = smp.losses.FocalLoss(
            mode=mode, ignore_index=ignore_index, normalized=normalized
        )

    def forward(self, preds, targets):
        return self.focal_loss(preds, targets)

# overall IOU for positive classes
def iou_test(pred,gt,epsilon=1e-6):
    iou = 0
    for b in range(gt.shape[0]):
        inter = torch.sum(((pred[b]==1) & (gt[b]==1)) | ((pred[b]==2) & (gt[b]==2)))
        union = torch.sum((pred[b]>0) | (gt[b]>0))
        iou += (inter + epsilon) / (union + epsilon)
    return iou/gt.shape[0]


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser('IAAI', add_help=False)
parser.add_argument('--root_dir',type=str)
parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/')
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--pretrained', type=str, default='')
parser.add_argument('--architecture',type=str,default='unet')
parser.add_argument('--backbone', type=str, default='resnet50')
parser.add_argument('--pretrain_type', type=str, default='imagenet')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
parser.add_argument('--subset', type=str, default='all')
parser.add_argument('--seed', type=int, default=42)

args = parser.parse_args()    


''' prepare checkpoint/log dir '''
os.makedirs(args.checkpoints_dir,exist_ok=True)
os.makedirs(os.path.join(args.checkpoints_dir,'val'),exist_ok=True)
tb_writer = SummaryWriter(os.path.join(args.checkpoints_dir,'log'))


''' prepare dataset and dataloader '''
transforms_train = K.AugmentationSequential(
    K.RandomResizedCrop((512, 512), scale=(0.5, 1.0), p=1.0),
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.5),
    K.RandomRotation(degrees=45,resample=Resample.NEAREST, p=0.5),
    K.RandomAffine((-15., 20.), resample=Resample.NEAREST, p=0.5),
    data_keys=["input", "mask"],
)

transforms_color = K.AugmentationSequential(
    #K.ColorJitter(0.4,0.4,0.4,0.4,p=0.5),
    #K.RandomGrayscale(p=0.5),
    K.RandomGaussianBlur((3,3),(0.1,2.0),p=0.01),
)

dataset = IAAI_dataset(args.root_dir,subset=args.subset,transforms=None,split='train')

train_data, val_data, _ = dataset_split(dataset, val_pct=0.5, test_pct=0.0, seed=args.seed)
train_dataset = Subset(dataset,transforms=transforms_train,transforms_color=transforms_color)
val_dataset = Subset(val_data,transforms=None)

train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,num_workers=args.num_workers,shuffle=True,pin_memory=True, drop_last=False)
val_dataloader = torch.utils.data.DataLoader(val_dataset,batch_size=1,shuffle=False,num_workers=args.num_workers,sampler=None,pin_memory=False, drop_last=False)

print('train_len: %d val_len: %d' % (len(train_dataset),len(val_dataset)))


''' prepare model '''
if args.subset=='all':
    in_channels = 11
elif args.subset=='rgb':
    in_channels = 3
    
# https://github.com/qubvel/segmentation_models.pytorch
if args.architecture=='unet':
    model = smp.Unet(
        encoder_name=args.backbone,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=args.pretrain_type,           # use `imagenet` pre-trained weights for encoder initialization
        in_channels=in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=3,                      # model output channels (number of classes in your dataset)
        decoder_attention_type='scse', # decoder attention type: spatial attention and channel attention
    )
if args.architecture=='deeplabv3+':
    model = smp.DeepLabV3Plus(
        encoder_name=args.backbone,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=args.pretrain_type,           # use `imagenet` pre-trained weights for encoder initialization
        in_channels=in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=3,                      # model output channels (number of classes in your dataset)
    )

if args.pretrained is not None:
  if os.path.isfile(args.pretrained):
    print('loading externel pretrained models requires customized codes here.')


''' define loss and optimizer '''
criterion_ce = torch.nn.CrossEntropyLoss()
criterion_dice = DiceLoss(mode="multiclass",ignore_index=0) # ignore the background class, note this can make the training more unstable

optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr)


''' optionally resume from interuption '''
last_epoch = 0
if args.resume:
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    last_epoch = checkpoint['epoch']
    last_loss = checkpoint['loss']

model.cuda() # the code is designed for single GPU


''' start training '''
print('Start training...')
for epoch in range(last_epoch,args.epochs):
    model.train()
    adjust_learning_rate(optimizer, epoch, args)
    
    loss_train = 0 # loss
    miou_train = 0 # mean intersection over union
    acc_train = 0 # overall accuracy
    oiou2_train = 0 # overall IOU of the two foreground classes (this is or is similar to the final evaluation metric)
    count_train = 0

    for i, data in enumerate(train_dataloader):
        inputs, targets = data[0].cuda(), data[1].cuda()
        
        output = model(inputs)
        
        loss = (criterion_ce(output, targets) + criterion_dice(output, targets))/2.0
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_miou = torchmetrics.functional.jaccard_index(torch.argmax(output,1).detach(), targets.detach(),ignore_index=0)
        train_acc = torchmetrics.functional.accuracy(torch.argmax(output,1).detach(), targets.detach())
        train_oiou2 = iou_test(torch.argmax(output,1).detach(),targets.detach())        
        
        loss_train += loss.item()
        miou_train += train_miou
        acc_train += train_acc
        oiou2_train += train_oiou2
        count_train += 1
                
        if i%9==0:
            print('Epoch {} Iter {}: train_loss {:.4f}, train_miou {:.3f}, train_acc {:.3f}, train_oiou2 {:.3f}'.format(epoch,i,loss_train/count_train,miou_train/count_train,acc_train/count_train,oiou2_train/count_train))
    
    # monitor training logs every epoch using tensorboard
    train_stats = {'loss': loss_train/count_train,
                   'miou': miou_train/count_train,
                   'acc' : acc_train/count_train,
                   'oiou2': oiou2_train/count_train}
    tb_writer.add_scalars('train', train_stats, global_step=epoch+1, walltime=None)

    # validate every 10 epochs
    if epoch%10==9:
        print('Validating...')
        model.eval()
        
        loss_val = 0
        miou_val = 0
        acc_val = 0
        oiou2_val = 0
        count_val = 0
        
        with torch.no_grad():
            for i, data in enumerate(val_dataloader):
                inputs, targets, ids = data[0].cuda(), data[1].cuda(), data[2]
                
                output = model(inputs)

                loss = (criterion_ce(output, targets) + criterion_dice(output, targets))/2.0

                val_miou = torchmetrics.functional.jaccard_index(torch.argmax(output,1).detach(),targets.detach(),ignore_index=0)
                val_acc = torchmetrics.functional.accuracy(torch.argmax(output,1).detach(), targets.detach())
                val_oiou2 = iou_test(torch.argmax(output,1).detach(),targets.detach())

                loss_val += loss.item()
                miou_val += val_miou
                acc_val += val_acc
                oiou2_val += val_oiou2            
                count_val += 1
                
                # predict some images for visual check
                if i%args.batch_size==0:
                    prediction = torch.argmax(output,1).detach().cpu().numpy()
                    cv2.imwrite(os.path.join(args.checkpoints_dir,'val','{}_ep{}.png'.format(ids[0],epoch)),prediction[0]*127)
        
        # monitor validation logs every 10 epoch using tensorboard
        val_stats = {'loss': loss_val/count_val,
                    'miou': miou_val/count_val,
                    'acc' : acc_val/count_val,
                    'oiou2': oiou2_val/count_val}
        tb_writer.add_scalars('val', val_stats, global_step=epoch+1, walltime=None)

        print('Epoch {}: val_loss {:.4f}, val_miou {:.3f}, val_acc {:.3f}, val_oiou2 {:.3f}'.format(epoch,val_stats['loss'],val_stats['miou'],val_stats['acc'],val_stats['oiou2']))
        
    # save the model every 10 epochs
    if epoch % 10 == 9:
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'loss':loss,
                    }, os.path.join(args.checkpoints_dir,'checkpoint_{:04d}.pth.tar'.format(epoch)))
    