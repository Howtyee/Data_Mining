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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('[Info]: use:', device)

Data_dir = './Datas/Datas.csv'
label_dir = './Datas/labels.csv'

SAVEDIR = './runs'

data = csv_dataset(Data_dir, label_dir)
train_set, test_set = random_split(data, [int(data.len * 0.8), data.len - int(data.len * 0.8)])



train_loader = DataLoader(dataset=train_set,
                          batch_size=8,
                          shuffle=True, )

test_loader = DataLoader(dataset=test_set,
                         batch_size=8,
                         shuffle=True, )

model = MLPClassifier()
model = model.to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_list = []
TP, TN, FP, FN = 0, 0, 0, 0
for epoch in range(10):
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
print(TP, TN, FP, FN)

acc = (TP + TN) / (TP + TN + FP + FN)
sen = (TP) / (FN + TP)
pre = (TP) / (TP + FP)

print('acc:', acc)
print('sen:', sen)
print('pre:', pre)
plt.plot(np.linspace(0, 100, len(loss_list)), loss_list)
plt.show()
