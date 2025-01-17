import os
import numpy as np
import torch
import linecache
import torch.utils.data as data
import random
import sys

class TestDataset(data.Dataset):
    def __init__(self, index, data_root=os.getcwd(), scaled=True, type='test'):
        super(TestDataset, self).__init__()
        
        self.type = type
        self.data = []
        if scaled:
            if self.type == 'train':
                dataset_name = 'dataset_scaled_train.csv'
            elif self.type == 'val':
                dataset_name = 'dataset_scaled_val.csv'
            elif self.type == 'test':
               dataset_name = 'dataset_scaled_test.csv'
            else:
                print('Invalid input')
                sys.exit()
        else:
            if self.type == 'train':
                dataset_name = 'dataset_train.csv'
            elif self.type == 'val':
                dataset_name = 'dataset_val.csv'
            elif self.type == 'test':
               dataset_name = 'dataset_test.csv'
            else:
                print('Invalid input')
                sys.exit()
        line = linecache.getline(data_root+'/data/'+dataset_name, index)
        # print(line)
        self.data.append(line)
        # print(data)

    def __getitem__(self, index):
        line = self.data[0]
        # print(line)
        line = line.split(',')
        label = torch.tensor(int(line[0]))
        dep = np.array(line[1:], dtype='float64').reshape(20,20)
        input_tensor =  torch.from_numpy(dep)
        return label, input_tensor

    def __len__(self):
        return len(self.data)