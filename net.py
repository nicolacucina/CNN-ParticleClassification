import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    # TODO: 2 output classes, requires CrossEntropyLoss
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )            
        self.max_pool = nn.MaxPool2d(2)
        self.fc1 = nn.Sequential(
            nn.Linear(102400, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.fc2 = nn.Linear(512, 2)
        # se intendo utilizzare 2 neuroni di uscita devo utilizzare CrossEntropyLoss, dove è già inclusa la funzione softmax, 
        # altrimenti se utilizzo 1 neurone di uscita devo utilizzare BCELosswithLogits, dove è già inclusa la funzione sigmoid
        # self.fc = nn.Linear(20 * 20 * 16, 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.float()
        # print('Dataset-x-unsqueeze: '+str(x))
        # print('Dataset-x-unsqueeze-shape: '+str(x.shape))
        # print('Dataset-x-unsqueeze-type: '+str(type(x)))
        
        out = self.conv1(x)
        # out = self.max_pool(out)
        out = self.conv2(out)
        out = self.max_pool(out)
        out = self.conv3(out)
        # out = self.max_pool(out)
        out = self.conv4(out)
        # out = self.max_pool(out)
        out = self.conv5(out)
        # out = self.max_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
            

class Net1(nn.Module):
    # 1 output class, requires BCEWithLogitsLoss
    def __init__(self, initial_out_channels=64):
        super(Net1, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            nn.ReLU()
        )
        
        self.max_pool = nn.MaxPool2d(2)
        
        self.block1 = self._make_block(64, 128, 2)
        self.block2 = self._make_block(128, 256, 2)
        self.block3 = self._make_block(256, 512, 2)

        f1_input_size = 512 * (20 // 2) * (20 // 2) # num layer max_pool
        
        self.fc1 = nn.Sequential(
            nn.Linear(f1_input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.fc2 = nn.Linear(512, 1)

    def _make_block(self, in_channels, out_channels, num_layers=2):
        layers = list()
        for i in range(num_layers):
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=(1,1), stride=(1,1)),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ]
            in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.float()
        
        out = self.conv1(x)
        # out = self.max_pool(out)

        out = self.block1(out)
        out = self.max_pool(out)

        out = self.block2(out)
        # out = self.max_pool(out)

        out = self.block3(out)
        # out = self.max_pool(out)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)

        return out