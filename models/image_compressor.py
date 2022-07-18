import torch
import torch.nn as nn
from modules import *


class ImageCompressor(nn.Module):

    def __init__(self, a, h, rank) -> None:
        super(ImageCompressor, self).__init__()
        self.a = a
        self.h = h
        self.device = torch.device('cuda:{:d}'.format(rank))
        self.encoder = Analysis_Net()
        self.decoder = Synthesis_Net()
        self.bit_estimator = BitsEstimator(192, K=5)
        self.entropy_coder = EntropyCoder(self.bit_estimator)

    def forward(self, inputs):
        """
        :param inputs: mini-batch
        :return: rec_imgs: 重构图像  bits_map: 累计分布函数
        """
        y = self.encoder(inputs)
        y_hat = self.quantize(y, is_train=True)
        rec_imgs = torch.clamp(self.decoder(y_hat), 0, 1)

        # R loss
        total_bits = torch.sum(
            torch.clamp(
                (-torch.log(
                    self.bit_estimator(y_hat + 0.5) - self.bit_estimator(y_hat - 0.5) + 1e-6)) / torch.log(
                    torch.tensor(2.0)),
                0,
                50))
        img_shape = rec_imgs.size()
        bpp = total_bits / (img_shape[0] * img_shape[2] * img_shape[3])
        # D loss
        distortion = torch.mean((inputs - rec_imgs) ** 2)
        # total loss
        loss = bpp + self.a.Lambda * (255 ** 2) * distortion

        return loss, bpp, distortion, rec_imgs

    def loss(self, inputs, loss_items):
        """
        :param inputs: original images
        :param loss_items: include bits_map and reconstruced images
        :param Lambda: trade-off
        :return:
        """
        y_hat, rec_imgs = loss_items
        # R loss
        total_bits = torch.sum(
            torch.clamp(
                (-torch.log(
                    self.bit_estimator(y_hat + 0.5) - self.bit_estimator(y_hat - 0.5) + 1e-6)) / torch.log(
                    torch.tensor(2.0)),
                0,
                50))
        img_shape = rec_imgs.size()
        bpp = total_bits / (img_shape[0] * img_shape[2] * img_shape[3])
        # D loss
        distortion = torch.mean((inputs - rec_imgs) ** 2)
        # total loss
        loss = bpp + self.a.Lambda * (255 ** 2) * distortion
        return loss, bpp, distortion

    def quantize(self, y, is_train=False):
        if is_train:
            uniform_noise = nn.init.uniform_(torch.zeros_like(y), -0.5, 0.5)
            if torch.cuda.is_available():
                uniform_noise = uniform_noise.to(self.device)
            y_hat = y + uniform_noise
        else:
            y_hat = torch.round(y)
        return y_hat

    def inference(self, img):
        """
        only use in test and validate
        """
        y = self.encoder(img)
        y_hat = self.quantize(y, is_train=False)
        stream, side_info = self.entropy_coder.compress(y_hat)
        y_hat_dec = self.entropy_coder.decompress(stream, side_info, y_hat.device)
        assert torch.equal(y_hat, y_hat_dec), "Entropy code decode for y_hat not consistent !"
        rec_img = torch.clamp(self.decoder(y_hat), 0, 1)
        bpp = len(stream) * 8 / img.shape[2] / img.shape[3]
        return rec_img, bpp
