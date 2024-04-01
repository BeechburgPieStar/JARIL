import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import time
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import time
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


batch_size = 512 


data_amp = sio.loadmat('data/test_data_split_amp.mat')
test_data_amp = data_amp['test_data']
test_data = test_data_amp

test_activity_label = data_amp['test_activity_label']
test_location_label = data_amp['test_location_label']
test_label = np.concatenate((test_activity_label, test_location_label), 1)

num_test_instances = len(test_data)

test_data = torch.from_numpy(test_data).type(torch.FloatTensor)
test_label = torch.from_numpy(test_label).type(torch.LongTensor)

test_dataset = TensorDataset(test_data, test_label)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model_name = 'XceptionTime_CSIMix_2.0'

aplnet = torch.load(f'weights/{model_name}.pkl')

aplnet = aplnet.cuda().eval()


correct_test_loc = 0
correct_test_act = 0
n_classes = 6
mean_tpr = 0.0
all_fpr = np.linspace(0, 1, 100)

for i, (samples, labels) in enumerate(test_data_loader):
    AR_target_output = []
    AR_pred_output = []
    IL_target_output = []
    IL_pred_output = []
    with torch.no_grad():
        samplesV = Variable(samples.cuda())
        labels_act = labels[:, 0].squeeze()
        labels_loc = labels[:, 1].squeeze()
        labelsV_act = Variable(labels_act.cuda())
        labelsV_loc = Variable(labels_loc.cuda())
        proba_act, proba_loc, feature = aplnet(samplesV)
        pred_act = proba_act.data.max(1)[1]
        correct_test_act += pred_act.eq(labelsV_act.data.long()).sum()
        pred_loc = proba_loc.data.max(1)[1]
        correct_test_loc += pred_loc.eq(labelsV_loc.data.long()).sum()

        print("AR Accuracy and IL Precision:")
        print(correct_test_act.cpu().numpy()/num_test_instances)
        print(correct_test_loc.cpu().numpy() / num_test_instances)


        AR_target_output[len(AR_target_output):len(labels_act)-1] = labels_act.tolist()
        IL_target_output[len(IL_target_output):len(labels_loc)-1] = labels_loc.tolist()

        print("Save Real and Predicted Location:")
        sio.savemat(f'vis/pred_{model_name}.mat', {'loc_prediction': pred_loc.cpu().numpy()})
        label = labelsV_loc.data.long()
        sio.savemat(f'vis/real_{model_name}.mat', {'loc_prediction': label.cpu().numpy()})

        print("Save AUC and ROC of AR:")
        AR_pred_score =  np.array(torch.Tensor.cpu(nn.Softmax(dim=1)(proba_act)))
        AR_target_output = np.array(AR_target_output)
        y_true_bin = label_binarize(AR_target_output, classes=np.arange(n_classes))       
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], AR_pred_score[:, i])
            print(np.shape(fpr))
            mean_tpr += np.interp(all_fpr, fpr, tpr)            
        mean_tpr /= n_classes
        roc_auc = auc(all_fpr, mean_tpr)
        sio.savemat(f'vis/AR_ROC_AUC={roc_auc}_{model_name}_all_fpr.mat',{'all_fpr':all_fpr})
        sio.savemat(f'vis/AR_ROC_AUC={roc_auc}_{model_name}_mean_tpr.mat',{'mean_tpr':mean_tpr})

        print("Print CM:")
        AR_pred_output[len(AR_pred_output):len(proba_act)-1] = (proba_act.argmax(dim=1, keepdim=True)).tolist()
        IL_pred_output[len(IL_pred_output):len(proba_loc)-1] = (proba_loc.argmax(dim=1, keepdim=True)).tolist()
        AR_pred_output = np.array(torch.Tensor.cpu(torch.Tensor(AR_pred_output)))
        AR_target_output = np.array(AR_target_output)
        IL_pred_output = np.array(torch.Tensor.cpu(torch.Tensor(IL_pred_output)))
        IL_target_output = np.array(IL_target_output)
        print(confusion_matrix(AR_target_output, AR_pred_output))
        print(confusion_matrix(IL_target_output, IL_pred_output))


