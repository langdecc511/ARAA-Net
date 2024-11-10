# ARAA-Net
  This is our research code of "Adaptive Region-Aware Attention Network for Epiphysis and Articular Surface Segmentation from Hand Radiograph"
  
  If you need any help for the code and data, do not hesitate to leave issues in this repository.
****
## Citation
 
```
@ARTICLE{10480742,
  author={Deng, Yamei and Song, Ting and Wang, Xu and Liao, Yong and Chen, Yonglu and He, Qian and Yao, Yun and Huang, Jianwei},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={ARAA-Net: Adaptive Region-Aware Attention Network for Epiphysis and Articular Surface Segmentation From Hand Radiographs}, 
  year={2024},
  volume={73},
  number={},
  pages={1-14},
  keywords={Bones;Feature extraction;Surface morphology;Adaptive systems;Convolution;Encoding;Data mining;Adaptive region-aware convolution (ARAC);articular surface segmentation;attention mechanism;epiphysis segmentation},
  doi={10.1109/TIM.2024.3381260}}


```
## Method
### Training
```

python train.py

```

### Testing

```

python test.py

```

## Notes

```

If you want to train the ARAANet on the TSRS_RSNA-Articular-Surface dataset, please update the "datasets_root" in the config.py. 

```
