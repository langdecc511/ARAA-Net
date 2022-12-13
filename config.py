#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 14:35:27 2021

@author: taiheng

"""
import os

backbone_path = './backbone/resnet/resnet50-19c8e357.pth'

datasets_root = './data/tooths'

cod_training_root = os.path.join(datasets_root, 'train')
chameleon_path = os.path.join(datasets_root, 'test')
