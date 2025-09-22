import torch.nn as nn
import torch.nn.functional as F
import torch

class RSM2D(nn.Module):
    def __init__(self, channels_in=None, channels_out=None):
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out

        self.conv1 = nn.Conv2d(in_channels=channels_in, out_channels=channels_out, bias=False, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=channels_out, out_channels=channels_out, bias=False, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=channels_out, out_channels=channels_out, bias=False, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(channels_out)
        self.bn2 = nn.BatchNorm2d(channels_out)
        self.bn3 = nn.BatchNorm2d(channels_out)

        self.nin = nn.Conv2d(in_channels=channels_in, out_channels=channels_out, bias=False, kernel_size=1)

    def forward(self, xx):
        yy = F.relu(self.bn1(self.conv1(xx)))
        yy = F.relu(self.bn2(self.conv2(yy)))
        yy = self.conv3(yy)
        xx = self.nin(xx)

        xx = self.bn3(xx + yy)
        xx = F.relu(xx)
        return xx
    
class SSDNet2D(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.4):
        super(SSDNet2D, self).__init__()
        num_channels = 32
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=num_channels, kernel_size=7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        
        self.RSM1 = RSM2D(channels_in=num_channels, channels_out=num_channels*2)
        self.RSM2 = RSM2D(channels_in=num_channels*2, channels_out=num_channels*4)
        self.RSM3 = RSM2D(channels_in=num_channels*4, channels_out=num_channels*8)
        self.RSM4 = RSM2D(channels_in=num_channels*8, channels_out=num_channels*8)

        self.fc1 = nn.Linear(in_features=num_channels*8, out_features=num_channels*4)
        self.fc2 = nn.Linear(in_features=num_channels*4, out_features=num_channels*2)
        self.fc3 = nn.Linear(in_features=num_channels*2, out_features=num_classes)
        self.out = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2)

        x = self.RSM1(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.RSM2(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.RSM3(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.RSM4(x)

        pool_kernel_size = (x.size(2), x.size(3))
        x = F.avg_pool2d(x, kernel_size=pool_kernel_size)

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x) 
        x = F.relu(self.fc2(x))
        x = self.dropout(x) 
        x = self.out(self.fc3(x))

        return x
