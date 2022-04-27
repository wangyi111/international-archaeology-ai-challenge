'''
Inference with a single model.

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
parser.add_argument('--architecture',type=str,default='unet')
parser.add_argument('--backbone', type=str, default='resnet50')
parser.add_argument('--batch_size', type=int, default=0)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--resume', default='', type=str, help='path to trained checkpoint')
parser.add_argument('--subset', type=str, default='all')

args = parser.parse_args()    


''' prepare data '''
dataset = IAAI_dataset(args.root_dir,subset=args.subset,transforms=None,split='test')
train_data, val_data, _ = dataset_split(dataset, val_pct=0.2, test_pct=0.0, seed=42)

dataloader = torch.utils.data.DataLoader(val_data,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,sampler=None,pin_memory=False, drop_last=False)

print('test_len: %d' % (len(dataloader)))


''' prepare model '''
if args.subset=='all':
    in_channels = 11
elif args.subset=='rgb':
    in_channels = 3

## https://github.com/qubvel/segmentation_models.pytorch
if args.architecture=='unet':
    model = smp.Unet(
        encoder_name=args.backbone,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=None,           # use `imagenet` pre-trained weights for encoder initialization
        in_channels=in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=3,                      # model output channels (number of classes in your dataset)
        decoder_attention_type='scse',
    )
if args.architecture=='deeplabv3+':
    model = smp.DeepLabV3Plus(
        encoder_name=args.backbone,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=None,           # use `imagenet` pre-trained weights for encoder initialization
        in_channels=in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=3,                      # model output channels (number of classes in your dataset)
    )


''' load model weights '''
try:
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['model_state_dict'])
except:
    print('checkpoint load error!')
    raise NotImplementedError

model.cuda()

''' inference with a single model '''
os.makedirs(args.out_dir,exist_ok=True)

print('Start testing...')
model.eval()
with torch.no_grad():
    for i, data in enumerate(tqdm(dataloader)):
        inputs,idx = data[0].cuda(),data[1]
        output = model(inputs)

        prediction = torch.squeeze(torch.argmax(output,1)).detach().cpu().numpy()

        prediction[prediction==1] = 128
        prediction[prediction==2] = 255

        cv2.imwrite(os.path.join(args.out_dir, 'pred_{}.png'.format(idx[0])),prediction)
