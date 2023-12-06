import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import linecache

class Dataset(data.Dataset):
    def __init__(self):
        super(Dataset, self).__init__()

        self.dataset_path = os.path.join(os.getcwd(), 'data', 'dataset_csv')
        
        self.data = open(self.dataset_path, "r")
        self.target = []
        self.input = []
        for line in self.data:
            line = line.split(',')
            self.target.append(line[0])
            self.input.append(np.array(line[1:], dtype='float32'))

    def __getitem__(self, index):
        return self.target[index], self.input[index].reshape(20,20)

    def __len__(self):
        return len(self.data)
        
        