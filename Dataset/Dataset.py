import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

Data_path = '../Datas/Datas.csv'
label_path = '../Datas/labels.csv'

class csv_dataset(Dataset):

    def __init__(self,D_p,l_p,k,train,path_np):
        all_array = np.load(path_np)
        if train == True:
            if k == 1:
                self.array = all_array[1008:]
            elif k == 2:
                self.array = np.append(all_array[0:1007],all_array[2014:])
            elif k == 3:
                self.array = np.append(all_array[0:2014], all_array[3021:])
            elif k == 4:
                self.array = np.append(all_array[0:3021], all_array[4028:])
            elif k == 5:
                self.array = all_array[0:4028]
        else:
            if k == 1:
                self.array = all_array[0:1007]
            elif k == 2:
                self.array = all_array[1007:2014]
            elif k == 3:
                self.array = all_array[2014:3021]
            elif k == 4:
                self.array = all_array[3021:4028]
            elif k == 5:
                self.array = all_array[4028:]

        pd_data = pd.read_csv(D_p)
        pd_label = pd.read_csv(l_p)

        self.X = np.array(pd_data)
        self.y = np.array(pd_label)

        self.len = len(self.array)



    def __getitem__(self, index):
        return self.X[self.array[index]], self.y[self.array[index]]

    def __len__(self):
        return self.len

if __name__ == '__main__':
    dataset = csv_dataset(Data_path,label_path,k=1,train=True,path_np='../Datas/random_allnums.npy')
    t = DataLoader(dataset,batch_size=1)
    for i,y in t:
        print(i.shape, y.shape)
