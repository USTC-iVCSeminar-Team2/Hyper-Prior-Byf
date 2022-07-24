import torch.nn as nn
import torch
from .GDN import GDN
import math


class Analysis_Net(nn.Module):
    def __init__(self, num_channel_N=128, num_channel_M=192):
        super(Analysis_Net, self).__init__()
        self.num_channel_N = num_channel_N
        self.num_channel_M = num_channel_M
        self.build_net()

    def build_net(self):
        self.conv1 = nn.Conv2d(3, self.num_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2*(3+self.num_channel_N)/6)))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)

        self.conv2 = nn.Conv2d(self.num_channel_N, self.num_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)

        self.conv3 = nn.Conv2d(self.num_channel_N, self.num_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)

        self.conv4 = nn.Conv2d(self.num_channel_N, self.num_channel_M, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv4.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.conv4.bias.data, 0.01)

        self.GDN1 = GDN(self.num_channel_N)
        self.GDN2 = GDN(self.num_channel_N)
        self.GDN3 = GDN(self.num_channel_N)

    def forward(self, inputs):
        x1 = self.GDN1(self.conv1(inputs))
        x2 = self.GDN2(self.conv2(x1))
        x3 = self.GDN3(self.conv3(x2))
        y = self.conv4(x3)  # y in shape of [B,M,16,16]
        return y