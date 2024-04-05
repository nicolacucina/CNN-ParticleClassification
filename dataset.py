import os
import numpy as np
import torch
import linecache
import torch.utils.data as data
import random
import sys
from utils import splitTrainTest

class Dataset(data.Dataset):
    def __init__(self, data_root=os.getcwd(), type='train'):
        super(Dataset, self).__init__()

        self.type = type
        self.data = []

        if self.type == 'train':
            with open(os.path.join(data_root, 'data', 'dataset_csv_train')) as f:
                for line in f:
                    self.data.append(line)
        elif self.type == 'val':
            with open(os.path.join(data_root, 'data', 'dataset_csv_val')) as f:
                for line in f:
                    self.data.append(line)
        elif self.type == 'test':
            with open(os.path.join(data_root, 'data', 'dataset_csv_test')) as f:
                for line in f:
                    self.data.append(line)
        else:
            print('Invalid input')
            sys.exit()

    def __getitem__(self, index):
        line = self.data[index]
        line = line.split(',')
        label = torch.tensor(int(line[0]))
        dep = np.array(line[1:], dtype='float64').reshape(20,20)
        input_tensor =  torch.from_numpy(dep)
        return label, input_tensor

    def __len__(self):
        return len(self.data)

        """
    def __init__(self, train, data_root=os.getcwd()):
        super(Dataset, self).__init__() 
        
        self.train = train
        if self.train:
            self.dataset_path = os.path.join(data_root, 'data', 'dataset_csv_train')
        else:
            self.dataset_path = os.path.join(data_root, 'data', 'dataset_csv_test')

    def __getitem__(self, index):
        line = linecache.getline(self.dataset_path, index)
        linecache.clearcache()
        line = line.split(',')
        label = torch.tensor(int(line[0]))
        dep = np.array(line[1:], dtype='float64').reshape(20,20)
        input_tensor =  torch.from_numpy(dep)
        return label, input_tensor 

    def __len__(self):
        return len(open(self.dataset_path).readlines())
    """
        
if __name__=='__main__':
    splitTrainTest(os.getcwd())
    temp = sys.argv[1]
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

    if dataset.type == 'train':
        print('Training set dimension: ' + str(dataset.__len__()))
    elif dataset.type == 'val':
        print('Validation set dimension: ' + str(dataset.__len__()))
    elif dataset.type == 'test':
        print('Test set dimension: ' + str(dataset.__len__()))
        
    ## Test the __getitem__ method and the datatype returned ##
    print('/////Testing the __getitem__ method/////')
    target, data = dataset.__getitem__(index=random.randint(0, dataset.__len__()))
    print('target: electron' if target == '0' else 'target: proton')
    print('data:' + str(data))
    print('memory occupation: ' + str(data.element_size()) + ' bytes per element * ' + str(data.nelement()) + ' elements * ' + str(dataset.__len__()) + ' items in the dataset = ' + str(temp:= data.element_size()*data.nelement()*dataset.__len__()) + ' bytes (' + str(temp/1024**2) + ' MB)')
    print('data shape: ' + str(data.shape))
    print('data type: ' + str(data.dtype))