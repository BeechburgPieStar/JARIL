# JARIL
Regularized Multi-Label Learning Empowered Joint Activity Recognition and Indoor Localization with CSI Fingerprints

# The code for test is published!

### Requirement

torch                              1.11.0+cu113

torchaudio                         0.11.0+cu113

torchsummary                       1.5.1

torchvision                        0.12.0+cu113

Python                             3.8.5

### File directory description

```
filetree 
├── README.md
├── /data/
│  ├── test_data_split_amp.mat
|  └── train_data_split_amp.mat
├── models
│  ├── XceptionTime_model.py
|  └── layers.py
├── weights
|  └── XceptionTime_CSIMix_2.0.pkl
├── result
├── vis
├── train_CSIMix.py
└── test.py

```

### Performance

AR accuracy: 0.9172661870503597; AUC: 0.9919091183016525

IL precision: 0.9964028776978417

### Thanks


- [geekfeiw](https://github.com/geekfeiw/ARIL)
- [timeseriesAI](https://github.com/timeseriesAI/tsai)
