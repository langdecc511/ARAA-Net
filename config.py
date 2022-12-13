#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-12-13 09:54:12

@author: XuWang

"""
import os

backbone_path = './backbone/resnet/resnet50-19c8e357.pth'

datasets_root = './data/TSRS_RSNA-Epiphysis'

cod_training_root = os.path.join(datasets_root, 'train')
chameleon_path = os.path.join(datasets_root, 'test')
