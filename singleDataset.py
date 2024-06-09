import os
import numpy as np
import torch
import linecache
import torch.utils.data as data
import random
import sys
from utils import splitTrainTest

class SingleDataset(data.Dataset):
    def __init__(self, index, data_root=os.getcwd(), scaled=True, type='test'):
        super(SingleDataset, self).__init__()

        self.type = type
        self.data = []
        if scaled:
            if self.type == 'train':
                dataset_name = 'dataset_csv_scaled_train'
            elif self.type == 'val':
                dataset_name = 'dataset_csv_scaled_val'
            elif self.type == 'test':
                dataset_name = 'dataset_csv_scaled_test'
            else:
                print('Invalid input')
                sys.exit()
        else:
            if self.type == 'train':
                dataset_name = 'dataset_csv_train'
            elif self.type == 'val':
                dataset_name = 'dataset_csv_val'
            elif self.type == 'test':
                dataset_name = 'dataset_csv_test'
            else:
                print('Invalid input')
                sys.exit()
        
        line = linecache.getline(os.path.join(data_root, 'data', dataset_name), index)
        self.data.append(line)

    def __getitem__(self, index):
        line = self.data[0]
        line = line.split(',')
        label = torch.tensor(int(line[0]))
        dep = np.array(line[1:], dtype='float64').reshape(20,20)
        input_tensor =  torch.from_numpy(dep)
        return label, input_tensor

    def __len__(self):
        return len(self.data)