import torch
from torch import nn
from .GDN import SetMinBoundary


class EntropyModelGaussion(nn.Module):
    def __init__(self):
        super(EntropyModelGaussion, self).__init__()

    def forward(self, input_, sigma):
        assert input_.shape[0:3] == sigma.shape[0:3], "Shape dismatch between y and gaussian sigma"
        mu = torch.zeros_like(input_)
        sigma = torch.clamp(sigma, 1e-6, 1e6)
        gaussian = torch.distributions.normal.Normal(mu, sigma)  # construct a gauss distribution
        cumul = gaussian.cdf(input_)
        return cumul

    def likelihood(self, input_, sigma):
        likelihood_ = self.forward(input_ + 0.5, sigma) - self.forward(input_ - 0.5, sigma) + 1e-6
        return likelihood_