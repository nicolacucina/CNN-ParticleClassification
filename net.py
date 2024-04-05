import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, type):
        super(Net, self).__init__()

        self.type = type
        if self.type == 'convolutional':
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
            self.conv1_1 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
            # self.conv2 = nn.Sequential(
            #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)),
            #     nn.BatchNorm2d(128),
            #     nn.ReLU()
            # )
            # self.conv3 = nn.Sequential(
            #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            #     nn.BatchNorm2d(128),
            #     nn.ReLU()
            # )
            # self.conv4 = nn.Sequential(
            #     nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)),
            #     nn.BatchNorm2d(256),
            #     nn.ReLU()
            # )
            # self.conv5 = nn.Sequential(
            #     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            #     nn.BatchNorm2d(256),
            #     nn.ReLU()
            # )
            self.fc = nn.Linear(20 * 20 * 64, 2)
        if self.type == 'fully_connected':
            #TODO
            return

    def forward(self, x):
        """print('Dataset-x: '+str(x))
        print('Dataset-x-shape: '+str(x.shape))
        print('Dataset-x-type: '+str(type(x)))"""
        x = x.unsqueeze(1)
        x = x.float()
        """print('Dataset-x-unsqueeze: '+str(x))
        print('Dataset-x-unsqueeze-shape: '+str(x.shape))
        print('Dataset-x-unsqueeze-type: '+str(type(x)))"""
        if self.type == 'convolutional':
            out = self.conv1(x)
            out = self.conv1_1(out)
            # out = self.conv2(out)
            # out = self.conv3(out)
            # out = self.conv4(out)
            # out = self.conv5(out)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return out
              
        if self.type == 'fully_connected':
            return out