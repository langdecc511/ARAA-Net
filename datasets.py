#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-12-13 09:54:12

@author: XuWang

"""

import os
import os.path
import torch.utils.data as data
from PIL import Image
import cv2
from PIL import Image
import numpy as np



import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import custom_transforms as tr
from torchvision.datasets.folder import is_image_file


def make_dataset(root):
    # image_path = os.path.join(root, 'Image')
    mask_path = os.path.join(root, 'GT')
    image_path = root
    mask_path = root + '_labels'
    
    #Images' format is jpg, GTs' format is png
    img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('.jpg')]
    return [(os.path.join(image_path, img_name + '.jpg'), os.path.join(mask_path, img_name + '.png')) for img_name in img_list]


class ImageFolder(data.Dataset):
    # image and gt should be in the same folder and have same filename except extended name (jpg and png respectively)
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None, split='train'):
        self.root = root
        self.imgs = make_dataset(root)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.label_mapping = {-1: -1, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6:6, 7:7, 8:8, 9:9, 10:10,
                              11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16:16, 17:17, 18:18, 19:19, 20:20,
                              21: 21, 22: 22, 23: 23, 24: 24, 25: 25, 26:26, 27:27, 28:28, 29:29, 30:30}

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path)
        label = self.convert_label(target)
        
        # print(gt_path)
        # print(target.size)
        # print(label.size)

        # if self.joint_transform is not None:
        #     img, label = self.joint_transform(img, label)
        
        sample = {'image': img, 'label': label}
        if self.split == "train":
            return self.transform_tr(sample) 
        else:
            sample = self.transform_val(sample)
            sample['name'] = self.imgs[index]
            return sample
        
    
    
    def convert_label(self, label):
        label_rgb = np.array(label)
        label_index = np.full(label_rgb.shape[:2], 0, dtype='uint8')
        
        for k in range(1,30):
            label_index[label_rgb == k] = 1

        label_index = Image.fromarray(label_index, mode='P')

        return label_index

    def __len__(self):
        return len(self.imgs)
    
    
    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.FixedResize(576,896),
            # tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomCrop((576,576)),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)
 
    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            # tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.FixedResize(576,896),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)
