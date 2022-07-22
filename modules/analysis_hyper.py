import torch.nn as nn
import torch
import math


class Analysis_Hyper_Net(nn.Module):
    def __init__(self, num_channel_N=128, num_channel_M=192):
        super(Analysis_Hyper_Net, self).__init__()
        self.num_channel_N = num_channel_N
        self.num_channel_M = num_channel_M
        self.build_net()

    def build_net(self):
        self.conv1 = nn.Conv2d(self.num_channel_M, self.num_channel_N, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)

        self.conv2 = nn.Conv2d(self.num_channel_N, self.num_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)

        self.conv3 = nn.Conv2d(self.num_channel_N, self.num_channel_N, 5, stride=2, padding=2, bias=False)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, (math.sqrt(2)))

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, inputs):
        y1 = torch.abs(inputs)
        y2 = self.relu1(self.conv1(y1))
        y3 = self.relu2(self.conv2(y2))
        z = self.conv3(y3)
        return z