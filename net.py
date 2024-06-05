import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# controllare downscaling che non collassino su immagini 1x1, massimo sottocampionare 2 volte fino a 5x5
# dopo maxpooling o sottocampionamento rimettere layer conv. con stride 1 e padding 1
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
            # self.conv1_1 = nn.Sequential(
            #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            #     nn.BatchNorm2d(64),
            #     nn.ReLU()
            # )
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(128),
                nn.ReLU()
            )
            # self.conv3 = nn.Sequential(
            #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            #     nn.BatchNorm2d(128),
            #     nn.ReLU()
            # )
            self.conv4 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
            # self.conv5 = nn.Sequential(
            #     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            #     nn.BatchNorm2d(256),
            #     nn.ReLU()
            # )
            self.conv6 = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(512),
                nn.ReLU()
            )
            self.conv7 = nn.Sequential(
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
        # self.fc = nn.Linear(20 * 20 * 16, 2)
        if self.type == 'fully_connected':
            #TODO
            return

    def forward(self, x):
        # print('Dataset-x: '+str(x))
        # print('Dataset-x-shape: '+str(x.shape))
        # print('Dataset-x-type: '+str(type(x)))
        x = x.unsqueeze(1)
        x = x.float()
        # print('Dataset-x-unsqueeze: '+str(x))
        # print('Dataset-x-unsqueeze-shape: '+str(x.shape))
        # print('Dataset-x-unsqueeze-type: '+str(type(x)))
        if self.type == 'convolutional':
            # out = self.conv1(x)
            # out = self.conv1_1(out)
            # out = self.conv2(out)
            # out = self.conv3(out)
            # out = self.conv4(out)
            # out = self.conv5(out)
            # out = out.view(out.size(0), -1)
            # out = self.fc(out)

            out = self.conv1(x)
            # out = self.max_pool(out)

            out = self.conv2(out)
            out = self.max_pool(out)

            out = self.conv4(out)
            # out = self.max_pool(out)

            out = self.conv6(out)
            out = self.conv7(out)
            # out = self.max_pool(out)
            out = out.view(out.size(0), -1)
            out = self.fc1(out)
            out = self.fc2(out)

            return out
              
        if self.type == 'fully_connected':
            return out