import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import os
from Dataset.Dataset import csv_dataset
from module.model import MLPClassifier
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import numpy as np
import pandas as pd
import tqdm
from sklearn.model_selection import StratifiedKFold

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('[Info]: use:', device)

Data_dir = './Datas/Datas.csv'
label_dir = './Datas/labels.csv'
np_file_dir = './Datas/random_allnums.npy'
SAVEDIR = './runs'

ALL_ACC, ALL_SEN, ALL_PRE = 0, 0, 0
for K_flod in range(1, 6):

    train_set = csv_dataset(Data_dir, label_dir, k=K_flod, train=True, path_np=np_file_dir)
    test_set = csv_dataset(Data_dir, label_dir, k=K_flod, train=False, path_np=np_file_dir)

    train_loader = DataLoader(dataset=train_set,
                              batch_size=8,
                              shuffle=True, )

    test_loader = DataLoader(dataset=test_set,
                             batch_size=8,
                             shuffle=True, )

    model = MLPClassifier()
    model = model.to(device)
    weights = torch.FloatTensor([2])
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    loss_list = []
    TP, TN, FP, FN = 0, 0, 0, 0
    for epoch in range(15):
        bar = tqdm.tqdm(train_loader)
        for X, y in bar:
            bar.set_description("epoch: %s" % str(epoch))
            X = X.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            y_pred = model(X)
            loss = criterion(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.data.item())
            bar.set_postfix(loss=loss.data.item())

        torch.save(model.state_dict(), os.path.join(SAVEDIR, str(epoch)) + '.pt')

    model.eval()
    bar2 = tqdm.tqdm(test_loader)
    for X, y in bar2:
        X = X.to(device, dtype=torch.float32)
        # y = y.to(device, dtype=torch.float32)
        y_pred = model(X)
        y_pred = F.sigmoid(y_pred)
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
        y_pred = torch.Tensor.cpu(y_pred)

        sTP = np.array(((y == 1) & (y_pred == 1)).sum())
        sFN = np.array(((y == 1) & (y_pred == 0)).sum())
        sTN = np.array(((y == 0) & (y_pred == 0)).sum())
        sFP = np.array(((y == 0) & (y_pred == 1)).sum())

        TP = sTP + TP
        TN = sTN + TN
        FP = sFP + FP
        FN = sFN + FN
    print('TP:',TP)
    print('TN:',TN)
    print('FP:',FP)
    print('FN:',FN)

    acc = (TP + TN) / (TP + TN + FP + FN)
    sen = (TP) / (FN + TP)
    pre = (TP) / (TP + FP)

    print('flod:' + str(K_flod) + ' acc:', acc)
    print('flod:' + str(K_flod) + ' sen:', sen)
    print('flod:' + str(K_flod) + ' pre:', pre)
    ALL_ACC = ALL_ACC + acc
    ALL_SEN = ALL_SEN + sen
    ALL_PRE = ALL_PRE + pre
    plt.plot(np.linspace(0, 100, len(loss_list)), loss_list)
    plt.show()
    print('Flod:',K_flod,' Finish!')
    print('==========================')
print('ALL_ACC:', ALL_ACC / 5)
print('ALL_SEN:', ALL_SEN / 5)
print('ALL_PRE:', ALL_PRE / 5)
