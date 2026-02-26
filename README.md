# JARIL
Y. Wang, H. Zhao, T. Ohtsuki, H. Sari and G. Gui, "Regularized Multi-Label Learning Empowered Joint Activity Recognition and Indoor Localization with CSI Fingerprints," in IEEE Transactions on Wireless Communications, doi: 10.1109/TWC.2024.3447786. 

# Revision Notice: Due to errors in the previous MACs computation for some models, all affected results have been recalculated. The values presented in the following table should be regarded as the authoritative results.

| Model          | MACs (G)     | Params (M)  |
|---------------|--------------|-------------|
| Proposed      | 0.156845346  | 0.980136    |
| InceptionTime | 0.205816128  | 1.05103     |
| TS_ResNet     | 0.389016448  | 1.011222    |
| ResNet        | 0.030273536  | 3.490326    |
| ResNet_plus   | 0.031849472  | 4.277782    |
| LSTM          | 0.005899648  | 0.031638    |
| BiLSTM        | 0.012066816  | 0.061846    |

### Requirement

```
torch                              1.11.0+cu113

torchaudio                         0.11.0+cu113

torchsummary                       1.5.1

torchvision                        0.12.0+cu113

Python                             3.8.5
```

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

# License / 许可证

本项目基于自定义非商业许可证发布，禁止用于任何形式的商业用途。

This project is distributed under a custom non-commercial license. Any form of commercial use is prohibited.

### Thanks


- [geekfeiw](https://github.com/geekfeiw/ARIL)
- [timeseriesAI](https://github.com/timeseriesAI/tsai)
