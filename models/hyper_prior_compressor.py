import torch
import torch.nn as nn
from modules import *


class HyperPrior(nn.Module):
    def __init__(self, a, h, rank) -> None:
        super(HyperPrior, self).__init__()
        self.a = a
        self.h = h
        self.device = torch.device('cuda:{:d}'.format(rank))
        self.encoder = Analysis_Net(num_channel_N=128, num_channel_M=192)
        self.decoder = Synthesis_Net(num_channel_N=128, num_channel_M=192)
        self.encoder_hyper = Analysis_Hyper_Net(num_channel_N=128, num_channel_M=192)
        self.decoder_hyper = Synthesis_Hyper_Net(num_channel_N=128, num_channel_M=192)
        self.bit_estimator_hyper = BitsEstimator(128, K=4)
        #self.entropy_coder_hyper = EntropyCoder(self.bit_estimator)

    def forward(self, inputs):
        """
        :param inputs: mini-batch
        """
        img_shape = inputs.size()
        y = self.encoder(inputs)
        y_hat = self.quantize(y, is_train=True)
        rec_imgs = torch.clamp(self.decoder(y_hat), 0, 1)
        z = self.encoder_hyper(y)
        z_hat = self.quantize(z, is_train=True)
        sigma = self.decoder_hyper(z_hat)

        # D loss
        distortion = torch.mean((inputs - rec_imgs) ** 2)
        # R_z loss
        total_bits_z = torch.sum(torch.clamp(
            -torch.log(
                self.bit_estimator_hyper(z_hat + 0.5) - self.bit_estimator_hyper(z_hat - 0.5) + 1e-10) / torch.log(
                torch.tensor(2)), 0, 50))
        # R_y loss
        mu = torch.zeros_like(y_hat)
        sigma = torch.clamp(sigma, 1e-10, 1e10)
        gaussian = torch.distributions.normal.Normal(mu, sigma)  # construct a gauss distribution
        probs = gaussian.cdf(y_hat + 0.5) - gaussian.cdf(y_hat - 0.5) + 1e-10
        total_bits_y = torch.sum(torch.clamp(-torch.log(probs) / torch.log(torch.tensor(2)), 0, 50))
        # R loss
        total_bits = total_bits_y + total_bits_z
        bpp = total_bits / (img_shape[0] * img_shape[2] * img_shape[3])

        # total loss
        loss = bpp + self.a.Lambda * distortion

        return loss, bpp, distortion, rec_imgs

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
        #stream, side_info = self.entropy_coder.compress(y_hat)
        #y_hat_dec = self.entropy_coder.decompress(stream, side_info, y_hat.device)
        #assert torch.equal(y_hat, y_hat_dec), "Entropy code decode for y_hat not consistent !"
        rec_img = torch.clamp(self.decoder(y_hat), 0, 1)
        #bpp = len(stream) * 8 / img.shape[2] / img.shape[3]
        return rec_img#, bpp
