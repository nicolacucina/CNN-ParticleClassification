import os
import numpy as np
import torch
import linecache
import torch.utils.data as data
import random
import sys
from utils import splitTrainTest

class Dataset(data.Dataset):
    def __init__(self, data_root=os.getcwd(), scaled=False, type='train', seed=42):
        super(Dataset, self).__init__()

        self.type = type
        self.seed = seed
        self.data = []
        if scaled:
            if self.type == 'train':
                with open(os.path.join(data_root, 'data', 'dataset_scaled_train.csv')) as f:
                    for line in f:
                        self.data.append(line)
            elif self.type == 'val':
                with open(os.path.join(data_root, 'data', 'dataset_scaled_val.csv')) as f:
                    for line in f:
                        self.data.append(line)
            elif self.type == 'test':
                with open(os.path.join(data_root, 'data', 'dataset_scaled_test.csv')) as f:
                    for line in f:
                        self.data.append(line)
            else:
                print('Invalid input')
                sys.exit()
        else:
            if self.type == 'train':
                with open(os.path.join(data_root, 'data', 'dataset_train.csv')) as f:
                    for line in f:
                        self.data.append(line)
            elif self.type == 'val':
                with open(os.path.join(data_root, 'data', 'dataset_val.csv')) as f:
                    for line in f:
                        self.data.append(line)
            elif self.type == 'test':
                with open(os.path.join(data_root, 'data', 'dataset_test.csv')) as f:
                    for line in f:
                        self.data.append(line)
            else:
                print('Invalid input')
                sys.exit()
        
        random.seed(self.seed)
        random.shuffle(self.data)

    def __getitem__(self, index):
        line = self.data[index]
        line = line.split(',')
        label = torch.tensor(int(line[0]))
        dep = np.array(line[1:], dtype='float64').reshape(20,20)
        input_tensor =  torch.from_numpy(dep)
        return label, input_tensor

    def __len__(self):
        return len(self.data)
       
if __name__=='__main__':
    splitTrainTest(os.getcwd())
    temp = sys.argv[1] # train, val, test
    if temp == 'train':
        print('Training set selected')
    elif temp == 'val':
        print('Validation set selected')
    elif temp == 'test':
        print('Test set selected')
    else:
        print('Invalid input')
        sys.exit()

    dataset = Dataset(data_root=os.getcwd(), type=temp)
    print('Dataset loaded')
    for i in range(10):
        print(dataset.data[i][0: 10] + ' ... ' + dataset.data[i][-10: -1])
    
    if dataset.type == 'train':
        print('Training set dimension: ' + str(dataset.__len__()))
    elif dataset.type == 'val':
        print('Validation set dimension: ' + str(dataset.__len__()))
    elif dataset.type == 'test':
        print('Test set dimension: ' + str(dataset.__len__()))
        
    # Test the __getitem__ method and the datatype returned
    print('*****Testing the __getitem__ method*****')
    target, data = dataset.__getitem__(index=random.randint(0, dataset.__len__()))
    print('target: electron' if target == '0' else 'target: proton')
    print('data:' + str(data))
    print('memory occupation: ' + str(data.element_size()) + ' bytes per element * ' + str(data.nelement()) + ' elements * ' + str(dataset.__len__()) + ' items in the dataset = ' + str(temp:= data.element_size()*data.nelement()*dataset.__len__()) + ' bytes (' + str(temp/1024**2) + ' MB)')
    print('data shape: ' + str(data.shape))
    print('data type: ' + str(data.dtype))