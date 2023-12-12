import os
import numpy as np
import torch
import linecache
import torch.utils.data as data
import random
import sys
from utils import splitTrainTest

class Dataset(data.Dataset):
    def __init__(self, train, data_root=os.getcwd()):
        super(Dataset, self).__init__() 
        
        self.train = train
        if self.train:
            self.dataset_path = os.path.join(data_root, 'data', 'dataset_csv_train')
        else:
            self.dataset_path = os.path.join(data_root, 'data', 'dataset_csv_test')
        
    def __getitem__(self, index):
        line = linecache.getline(self.dataset_path, index)
        line = line.split(',')
        dep = np.array(line[1:], dtype='float64').reshape(20,20)
        input_tensor =  torch.from_numpy(dep)
        return line[0], input_tensor

    def __len__(self):
        return len(open(self.dataset_path).readlines())
        
if __name__=='__main__':
    splitTrainTest(os.getcwd())
    dataset = Dataset(data_root=os.getcwd(), train=False)
    print('Training set dimension: ' if dataset.train else 'Test set dimension: ' + str(dataset.__len__()))
    target, data = dataset.__getitem__(index=random.randint(0, dataset.__len__()))
    print('target: electron' if target == '0' else 'target: proton')
    print('data:')
    print(data)
    print('memory occupation: ' + str(data.element_size()) + ' bytes per element * ' + str(data.nelement()) + ' elements = ' + str(data.element_size()*data.nelement()) + ' bytes ')
    
    # this is the occupation of the pointer, not of the data
    #print('memory occupation: '+ str(sys.getsizeof(data))+ ' bytes')
    print('data shape: ' + str(data.shape))
    print('data type: ' + str(data.dtype))