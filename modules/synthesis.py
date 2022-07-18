import torch
import torch.nn as nn
import math
from .GDN import GDN


class Synthesis_Net(nn.Module):
    def __init__(self, num_channel_N=128, num_channel_M=192):
        super(Synthesis_Net, self).__init__()
        self.num_channel_N = num_channel_N
        self.num_channel_M = num_channel_M
        self.build_net()

    def build_net(self):
        self.deconv1 = nn.ConvTranspose2d(self.num_channel_M, self.num_channel_N, 5, stride=2, padding=2,
                                          output_padding=1, bias=False)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, (math.sqrt(2)))

        self.deconv2 = nn.ConvTranspose2d(self.num_channel_N, self.num_channel_N, 5, stride=2, padding=2,
                                          output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)

        self.deconv3 = nn.ConvTranspose2d(self.num_channel_N, self.num_channel_N, 5, stride=2, padding=2,
                                          output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)

        self.deconv4 = nn.ConvTranspose2d(self.num_channel_N, 3, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv4.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.deconv4.bias.data, 0.01)

        self.igdn1 = GDN(self.num_channel_N, inverse=True)
        self.igdn2 = GDN(self.num_channel_N, inverse=True)
        self.igdn3 = GDN(self.num_channel_N, inverse=True)

    def forward(self, inputs):
        x3 = self.igdn1(self.deconv1(inputs))
        x2 = self.igdn2(self.deconv2(x3))
        x1 = self.igdn3(self.deconv3(x2))
        x_hat = self.deconv4(x1)  # x_hat in shape of [B,3,256,256]
        return x_hat