#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-12-13 09:54:12

@author: XuWang

"""
import datetime
import time
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

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

import joint_transforms
from config import cod_training_root,chameleon_path
from config import backbone_path
from datasets import ImageFolder
from misc import AvgMeter, check_mkdir
from daseg import daseg
from dar import DARConv2d
import loss

from seg_utils import ConfusionMatrix

cudnn.benchmark = True

torch.manual_seed(2021)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")







ckpt_path = './ckpt'
exp_name = 'DANet'

args = {
    'epoch_num': 1000,
    'train_batch_size': 10,
    'last_epoch': 0,
    'lr': 1e-3,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': '',
    'scale_w': 576,
    'scale_h': 896,
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

# Transform Data.
joint_transform = joint_transforms.Compose([
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.Resize((args['scale_w'], args['scale_h'])),
    # joint_transforms.FixedResizewx(args['scale_w']),
    joint_transforms.RandomCrop([576, 576], pad_if_needed=True, lbl_fill=255)
])


joint_transform_val = joint_transforms.Compose([
    joint_transforms.Resize((args['scale_w'], args['scale_h']))
])


img_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()

# Prepare Data Set.
train_set = ImageFolder(cod_training_root, joint_transform, img_transform, target_transform)
print("Train set: {}".format(train_set.__len__()))
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=0, shuffle=True)


test_set = ImageFolder(chameleon_path, joint_transform_val, img_transform, target_transform)
print("Test set: {}".format(test_set.__len__()))
test_loader = DataLoader(test_set, batch_size=1, num_workers=0, shuffle=False)










total_epoch = args['epoch_num'] * len(train_loader)
print("total_epoch",total_epoch)

# loss function
structure_loss = loss.structure_loss().to(device)
bce_loss = nn.BCEWithLogitsLoss().to(device)
iou_loss = loss.IOU().to(device)
last_criterion = nn.CrossEntropyLoss(ignore_index=255)
print("iou_loss=",iou_loss)

def bce_iou_loss(pred, target):
    bce_out = bce_loss(pred, target)
    iou_out = iou_loss(pred, target)

    loss = bce_out + iou_out

    return loss

def train(net, optimizer):
    net.train()
    curr_iter = 1
    start_time = time.time()
    
    best_mIoU = 0
    for epoch in range(args['last_epoch'] + 1, args['last_epoch'] + 1 + args['epoch_num']):
        loss_record, loss_1_record, loss_2_record, loss_3_record, loss_4_record, loss_0_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

        train_iterator = tqdm(train_loader, total=len(train_loader))
        confmat = ConfusionMatrix(num_classes=2)
        for data in train_iterator:
            if args['poly_train']:
                base_lr = args['lr'] * (1 - float(curr_iter) / float(total_epoch)) ** args['lr_decay']
                optimizer.param_groups[0]['lr'] = 2 * base_lr
                optimizer.param_groups[1]['lr'] = 1 * base_lr



            inputs, labels = data['image'], data['label']
            batch_size = inputs.size(0)
            inputs = Variable(inputs).to(device)
            labels = Variable(labels).to(device)
            
            
            # from  torchvision import utils as vutils
            # iinputs = inputs[0].cpu().clone()
            # vutils.save_image(iinputs, './pred_input.jpg', normalize=True)
            
            # class_color = [(0, 0, 0),(128, 0, 0)]          
            # import numpy as np
            # lla = labels.long().cpu().clone()
            # lla = np.uint8(lla.squeeze(0).squeeze(0))
            # label_index = np.full(inputs.squeeze(0).shape, 255, dtype='uint8')
            # for ik, color in enumerate(class_color):
            #     for ig in range(label_index.shape[0]):
            #         label_index[ig][lla == ik] = color[ig]
                    
            # image = torch.from_numpy(label_index).float()
            # vutils.save_image(image, 'labels_input.jpg')

            optimizer.zero_grad()

            predict_1, predict_2, predict_3, predict_4, predict0 = net(inputs)

            loss_1 = bce_iou_loss(predict_1, labels.unsqueeze(1))
            loss_2 = structure_loss(predict_2, labels.unsqueeze(1))
            loss_3 = structure_loss(predict_3, labels.unsqueeze(1))
            loss_4 = structure_loss(predict_4, labels.unsqueeze(1))       
            loss_0 = last_criterion(predict0, labels.long())

            loss = 1 * loss_1 + 1 * loss_2 + 2 * loss_3 + 4 * loss_4 + 10*loss_0

            loss.backward()
            optimizer.step()

            loss_record.update(loss.data, batch_size)
            loss_1_record.update(loss_1.data, batch_size)
            loss_2_record.update(loss_2.data, batch_size)
            loss_3_record.update(loss_3.data, batch_size)
            loss_4_record.update(loss_4.data, batch_size)
            loss_0_record.update(loss_0.data, batch_size)
            
            
            pred = predict0.data.cpu().numpy()
            target = labels.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            
            
            confmat.update(labels.flatten(), predict0.argmax(1).flatten() if predict0.dim() == 4 else predict0.flatten())
        
        
            
            global_acc, class_acc, class_iou,FWIoU,mDice = confmat.compute()

            if curr_iter % 10 == 0:
                writer.add_scalar('loss', loss, curr_iter)
                writer.add_scalar('loss_1', loss_1, curr_iter)
                writer.add_scalar('loss_2', loss_2, curr_iter)
                writer.add_scalar('loss_3', loss_3, curr_iter)
                writer.add_scalar('loss_4', loss_4, curr_iter)
                writer.add_scalar('loss_0', loss_0, curr_iter)


            log = '[%3d], [%6d], [%.6f], [%.5f], [%.5f]' % \
                  (epoch, curr_iter, base_lr, np.mean(class_iou.cpu().numpy()), loss_0_record.avg)
            train_iterator.set_description(log)
            open(log_path, 'a').write(log + '\n')

            curr_iter += 1


        current_mIoU = validate(net, optimizer)

        if best_mIoU < current_mIoU:
            best_mIoU = current_mIoU
            net.cpu()
            torch.save(net.module.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % epoch))
            net.to(device)
            


def validate(net, optimizer):
    net.eval()
    curr_iter = 1
    start_time = time.time()

    confmat = ConfusionMatrix(num_classes=2)
    loss_record, loss_1_record, loss_2_record, loss_3_record, loss_4_record, loss_0_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

    test_iterator = tqdm(test_loader, total=len(test_loader))
    for data in test_iterator:
        inputs, labels = data['image'], data['label']
        
        
        # from  torchvision import utils as vutils
        # iinputs = inputs[0].cpu().clone()
        # vutils.save_image(iinputs, './pred_input.jpg', normalize=True)
        
        
        
        # class_color = [(0, 0, 0),(128, 0, 0)]          
        # import numpy as np
        # lla = labels.long().cpu().clone()
        # lla = np.uint8(lla.squeeze(0).squeeze(0))
        # label_index = np.full(inputs.squeeze(0).shape, 255, dtype='uint8')
        # for ik, color in enumerate(class_color):
        #     for ig in range(label_index.shape[0]):
        #         label_index[ig][lla == ik] = color[ig]
                
        # image = torch.from_numpy(label_index).float()
        # vutils.save_image(image, 'labels_input.jpg')
        
        
        batch_size = inputs.size(0)
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            predict_1, predict_2, predict_3, predict_4, predict0 = net(inputs)
    
            loss_1 = bce_iou_loss(predict_1, labels.unsqueeze(1))
            loss_2 = structure_loss(predict_2, labels.unsqueeze(1))
            loss_3 = structure_loss(predict_3, labels.unsqueeze(1))
            loss_4 = structure_loss(predict_4, labels.unsqueeze(1))
            
            loss_0 = last_criterion(predict0, labels.long())
            
    
            loss = 1 * loss_1 + 1 * loss_2 + 2 * loss_3 + 4 * loss_4 + 10*loss_0
    
            loss_record.update(loss.data, batch_size)
            loss_1_record.update(loss_1.data, batch_size)
            loss_2_record.update(loss_2.data, batch_size)
            loss_3_record.update(loss_3.data, batch_size)
            loss_4_record.update(loss_4.data, batch_size)
            loss_0_record.update(loss_0.data, batch_size)
            
            
            pred = predict0.data.cpu().numpy()
            target = labels.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            
            
            confmat.update(labels.flatten(), predict0.argmax(1).flatten() if predict0.dim() == 4 else predict0.flatten())
        
            

            global_acc, class_acc, class_iou,FWIoU,mDice = confmat.compute()
            class_iou = class_iou.cpu().numpy()
            
            log = '[%3d], [%6f], [%.5f]' % \
                  (curr_iter, np.mean(class_iou), loss_0_record.avg)
            test_iterator.set_description(log)
            open(log_path, 'a').write(log + '\n')
    
            curr_iter += 1
            # break


    global_acc, class_acc, class_iou,FWIoU = confmat.compute()
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
    # print("args=",args)
    net = daseg(backbone_path).train()
    net = net.to(device)

    if args['optimizer'] == 'Adam':
        print("Adam")
        optimizer = optim.Adam([
            {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
             'lr': 2 * args['lr']},
            {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
             'lr': 1 * args['lr'], 'weight_decay': args['weight_decay']}
        ])
    else:
        print("SGD")
        optimizer = optim.SGD([
            {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
             'lr': 2 * args['lr']},
            {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
             'lr': 1 * args['lr'], 'weight_decay': args['weight_decay']}
        ], momentum=args['momentum'])

    if len(args['snapshot']) > 0:
        print('Training Resumes From \'%s\'' % args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
        total_epoch = (args['epoch_num'] - int(args['snapshot'])) * len(train_loader)
        print(total_epoch)

    net = nn.DataParallel(net)
    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer)
    writer.close()


if __name__ == '__main__':
    main()
