#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 14:35:27 2021

@author: taiheng

"""
import time
import datetime
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from config import cod_training_root,chameleon_path

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
from torchvision import utils as vutils

import joint_transforms
from config import cod_training_root,chameleon_path
from config import backbone_path
from datasets import ImageFolder
from misc import AvgMeter, check_mkdir
from daseg import daseg
from dar import DARConv2d
import loss
import numpy as np
import matplotlib.pyplot as plt

from seg_utils import ConfusionMatrix

cudnn.benchmark = True





import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from collections import OrderedDict
from numpy import mean

from config import *
from misc import *
from daseg import daseg

torch.manual_seed(2021)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


ckpt_path = './ckpt'
exp_name = 'TASNet'

args = {
    'epoch_num': 2000,
    'train_batch_size': 2,
    'last_epoch': 0,
    'lr': 5e-4,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': '',
    'scale_w': 576,
    'scale_h': 896,
    'save_point': [0,50,80,100,120,135,180,260,300,500,700,900,1000,1200,1300,1400,1500,1600,1700,1800,1900,1999],
    'poly_train': True,
    'optimizer': 'Adam',
}

# Path.
check_mkdir(ckpt_path)
check_mkdir(os.path.join(ckpt_path, exp_name))
vis_path = os.path.join(ckpt_path, exp_name, 'log')
check_mkdir(vis_path)
log_path = os.path.join(ckpt_path, exp_name,'log.txt')
writer = SummaryWriter(log_dir=vis_path, comment=exp_name)


results_path = './results'
check_mkdir(results_path)
exp_name = 'TASNet'
args = {
    'scale_w': 576,
    'scale_h': 896,
    'save_results': True
}

print(torch.__version__)

# img_transform = transforms.Compose([
#     transforms.Resize((args['scale_w'], args['scale_h']))
# ])

joint_transform_val = joint_transforms.Compose([
    joint_transforms.Resize((args['scale_w'], args['scale_h']))
])


img_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()

to_pil = transforms.ToPILImage()


to_test = OrderedDict([
                       ('DAGM', chameleon_path)
                       ])
results = OrderedDict()

structure_loss = loss.structure_loss().to(device)
bce_loss = nn.BCEWithLogitsLoss().to(device)
iou_loss = loss.IOU().to(device)
last_criterion = nn.CrossEntropyLoss(ignore_index=255)

def bce_iou_loss(pred, target):
    bce_out = bce_loss(pred, target)
    iou_out = iou_loss(pred, target)

    loss = bce_out + iou_out

    return loss



test_set = ImageFolder(chameleon_path, joint_transform_val, img_transform, target_transform,split='test')
print("Test set: {}".format(test_set.__len__()))
test_loader = DataLoader(test_set, batch_size=1, num_workers=0, shuffle=False)




def sample_images(epoch, batch_i, MECG, FECG_reconstr, FECG, sample_path):

    r, c = 1, 3
    gen_imgs = [MECG, FECG_reconstr,FECG]
    titles = ['Image', 'Image_rec','Label']
    
    fig, axs = plt.subplots(r, c,figsize=(15, 5))
    cnt = 0
    for i in range(r):
        for j in range(c):
            for bias in range(1):
                tt = gen_imgs[cnt]
                A = tt.cpu().detach().numpy()
                A = A[bias].squeeze(0)
                axs[j].imshow(A)
            axs[j].set_title(titles[j])
            cnt += 1
    fig.savefig("%s/%d_%d.png" % (sample_path, epoch,batch_i),dpi=500,bbox_inches = 'tight')
    plt.close()


def validate(net):
    net.eval()
    curr_iter = 1
    start_time = time.time()

    confmat = ConfusionMatrix(num_classes=2)
    loss_record, loss_1_record, loss_2_record, loss_3_record, loss_4_record, loss_0_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

    test_iterator = tqdm(test_loader, total=len(test_loader))
    for data in test_iterator:

        inputs, labels,name = data['image'], data['label'],  data['name']
        
        batch_size = inputs.size(0)
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            predict_1, predict_2, predict_3, predict_4, predict0 = net(inputs)
            
            
            pred = predict0.argmax(1)
            # target = labels.cpu().numpy()
            # pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            
            
            confmat.update(labels.flatten(), predict0.argmax(1).flatten() if predict0.dim() == 4 else predict0.flatten())
        
            class_color = [(0, 0, 0),(255, 255, 255)]          
            lla = pred.long().cpu().clone()
            lla = np.uint8(lla.squeeze(0).squeeze(0))
            label_index = np.full(inputs.squeeze(0).shape, 255, dtype='uint8')
            for ik, color in enumerate(class_color):
                for ig in range(label_index.shape[0]):
                    label_index[ig][lla == ik] = color[ig]
                    
            image = torch.from_numpy(label_index).float()
            pname = name[0][0].split('/')[-1]
            path = 'data/segment/results/' + pname
            vutils.save_image(image, path)    
        
        
            # sample_images(1, curr_iter, inputs, pred, labels, './results/TASNet/DAGM')
        

            global_acc, class_acc, class_iou,FWIoU,mDice = confmat.compute()
            class_iou = class_iou.cpu().numpy()
            
            log = '[%3d], [%6f], [%.5f]' % \
                  (curr_iter, np.mean(class_iou), loss_0_record.avg)
            test_iterator.set_description(log)
            open(log_path, 'a').write(log + '\n')
    
            curr_iter += 1


    global_acc, class_acc, class_iou,FWIoU,mDice = confmat.compute()
    global_acc = global_acc.item()
    class_acc = class_acc.cpu().numpy()
    class_iou = class_iou.cpu().numpy()
    FWIoU = FWIoU.cpu().numpy()
    
    
    print(f'global_acc={global_acc}')
    print(f'class_acc={class_acc}')
    print(f'class_iou={class_iou}')
    print(f'mIoU={np.mean(class_iou)}')
    print(f'FWIoU={FWIoU}')
    print(f'mDice={mDice}')
    
    return np.mean(class_iou)





def main():
    net = daseg(backbone_path).to(device)

    #net.load_state_dict(torch.load('TASNet.pth'))
    net.load_state_dict(torch.load('ckpt/TASNet/223.pth'))

    net.eval()
    with torch.no_grad():
        start = time.time()
        validate(net)

    end = time.time()
    print("Total Testing Time: {}".format(str(datetime.timedelta(seconds=int(end - start)))))

if __name__ == '__main__':
    main()
