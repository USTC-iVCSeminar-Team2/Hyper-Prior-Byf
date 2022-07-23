import torch
from torch import nn
import math


class Synthesis_Hyper_Net(nn.Module):
    def __init__(self, num_channel_N=128, num_channel_M=192):
        super(Synthesis_Hyper_Net, self).__init__()
        self.num_channel_N = num_channel_N
        self.num_channel_M = num_channel_M
        self.build_net()

    def build_net(self):
        self.deconv1 = nn.ConvTranspose2d(self.num_channel_N, self.num_channel_N, 5, 2, 2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.deconv2 = nn.ConvTranspose2d(self.num_channel_N, self.num_channel_N, 5, 2, 2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.deconv3 = nn.ConvTranspose2d(self.num_channel_N, self.num_channel_M, 3, 1, 1)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, (
            math.sqrt(2 * (self.num_channel_N + self.num_channel_M) / (self.num_channel_N + self.num_channel_N))))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, inputs):
        y2 = self.relu1(self.deconv1(inputs))
        y1 = self.relu2(self.deconv2(y2))
        y_hat = self.relu3(self.deconv3(y1))
        return y_hat
