import torch
import torch.nn as nn
from modules import *


class HyperPrior(nn.Module):
    def __init__(self, a, h, rank, N=128, M=192) -> None:
        super(HyperPrior, self).__init__()
        self.a = a
        self.h = h
        self.device = torch.device('cuda:{:d}'.format(rank))
        self.encoder = Analysis_Net(num_channel_N=N, num_channel_M=M)
        self.decoder = Synthesis_Net(num_channel_N=N, num_channel_M=M)
        self.encoder_hyper = Analysis_Hyper_Net(num_channel_N=N, num_channel_M=M)
        self.decoder_hyper = Synthesis_Hyper_Net(num_channel_N=N, num_channel_M=M)
        self.bit_estimator_hyper = BitsEstimator(N, K=4)
        self.entropy_coder_hyper = EntropyCoder(self.bit_estimator_hyper)
        self.entropy_model_gaussion = EntropyModelGaussion()
        self.entropy_coder_gaussion = EntropyCoderGaussian(self.entropy_model_gaussion)

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
        distortion = torch.mean((inputs - rec_imgs).pow(2))
        # R_z loss
        total_bits_z = torch.sum(torch.clamp(-1.0 * torch.log2(self.bit_estimator_hyper.likelihood(z_hat)), 0, 50))
        # R_y loss
        total_bits_y = torch.sum(torch.clamp(-1.0 * torch.log2(self.entropy_model_gaussion.likelihood(y_hat, sigma)),
                                             0, 50))
        # R loss
        bpp_y = total_bits_y / (img_shape[0] * img_shape[2] * img_shape[3])
        bpp_z = total_bits_z / (img_shape[0] * img_shape[2] * img_shape[3])
        bpp = bpp_y + bpp_z

        # total loss
        loss = bpp + self.a.Lambda * distortion * 255 ** 2

        return loss, bpp, bpp_y, bpp_z, distortion, rec_imgs

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
        # stream, side_info = self.entropy_coder.compress(y_hat)
        # y_hat_dec = self.entropy_coder.decompress(stream, side_info, y_hat.device)
        # assert torch.equal(y_hat, y_hat_dec), "Entropy code decode for y_hat not consistent !"
        rec_img = torch.clamp(self.decoder(y_hat), 0, 1)
        # bpp = len(stream) * 8 / img.shape[2] / img.shape[3]
        return rec_img  # , bpp
