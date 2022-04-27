'''
Test and evaluate the score.
2022.04.27
'''

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import jaccard_score,accuracy_score
import torch
import torch.nn.functional as F
import pdb

# overall IOU for positive classes
def iou_test(pred,gt,epsilon=1e-6):
    inter = np.sum(((pred==128) & (gt==128)) | ((pred==255) & (gt==255)))
    union = np.sum((pred>0) | (gt>0))
    return (inter + epsilon) / (union + epsilon)

pred_dir = os.path.join('checkpoints','merge_model','test')
pred_files = os.listdir(pred_dir)
gt_dir = os.path.join('data','Hackathon_data','Ground_truth')

miou = 0
acc = 0
iou2 = 0
count = 0
for pred_file in tqdm(pred_files):
    pred_path = os.path.join(pred_dir,pred_file)
    gt_file = pred_file.replace('pred','mask')
    gt_path = os.path.join(gt_dir,gt_file)

    mask = np.asarray(Image.open(gt_path))
    pred = np.asarray(Image.open(pred_path))

    miou += jaccard_score(mask.flatten(),pred.flatten(),average='macro') # mean IOU averaged over each class
    iou2 += iou_test(pred,mask) # overall IOU for positive classes

    count += 1

print('miou {}, iou2 {}'.format(miou/count,iou2/count))


