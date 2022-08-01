import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function


class SetMinBoundary(Function):
    """
    Set parameter in GDN to min boundary after each gradient step which is 2^-5 in the paper.
    """

    @staticmethod
    def forward(ctx, input, min_boundary):
        b = torch.ones_like(input) * min_boundary
        ctx.save_for_backward(input, b)
        return torch.max(input, b)

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param grad_output: gradient from previous layer
        :return: grandient
        """
        input, b = ctx.saved_tensors
        passthrough_1 = input >= b
        passthrough_2 = grad_output < 0
        passthrough_map = passthrough_1 | passthrough_2
        return passthrough_map.type(grad_output.dtype) * grad_output, None


class GDN(nn.Module):
    def __init__(self, num_output_channel, beta_min=1e-6, beta_init=0.1, gamma_min=1e-6, gamma_init=0.1,
                 min_boundary=2 ** -5, inverse=False):
        """
        :param beta_min: a small positive value to ensure beta' in range(2e-5,...)
        :param gamma_init: gamma initiated value
        :param num_output_channel: It is same for in/out because it is only a 'nomalization'
        :param min_boundary: the lower boundary for 'gamma' and 'beta''
        :param inverse: Identify GDN or IGDN
        """
        super(GDN, self).__init__()
        self.min_boundary = min_boundary
        self.inverse = inverse
        self.num_output_channel = num_output_channel
        self.reparam_offset = min_boundary ** 2
        self.beta_bound = (beta_min + self.reparam_offset) ** 0.5
        self.gamma_bound = (gamma_min + self.reparam_offset) ** 0.5
        # cos arrange
        # self.I = torch.arange(self.num_output_channel).repeat(self.num_output_channel)
        # self.J = torch.einsum('ij->ji',
        #                       torch.arange(self.num_output_channel).repeat((self.num_output_channel, 1))).reshape(
        #     -1)
        # beta, gamma
        self.beta = nn.Parameter(torch.sqrt(torch.ones(num_output_channel) * beta_init + self.reparam_offset))
        self.gamma = nn.Parameter(torch.sqrt(torch.eye(num_output_channel) * gamma_init + self.reparam_offset))

    def forward(self, inputs):
        B, C, H, W = inputs.size()
        # transpose average
        gamma_T = self.gamma.transpose(0, 1)
        gamma_p = (self.gamma + gamma_T) / 2

        # lower boundary
        beta_p = SetMinBoundary.apply(self.beta, self.beta_bound)
        beta = beta_p ** 2 - self.reparam_offset

        gamma_p = SetMinBoundary.apply(gamma_p, self.gamma_bound)
        gamma = gamma_p ** 2 - self.reparam_offset

        # with torch.no_grad():
        #     X = torch.einsum('qwer->wqer', inputs)
        #     X = X.reshape(C, B * H * W)
        #     A = torch.einsum('ij,jk->ik', X, X.permute(1, 0))
        #     norm = torch.norm(X, p=2, dim=1).unsqueeze(0)
        #     B = torch.einsum('ij,jk->ik', norm.permute(1, 0), norm)
        #     cos_similarities = A / (B + 1e-6)
        #
        # weight = gamma * (1 + cos_similarities)/2
        # # tensor转化为一维
        # weight = weight.reshape(self.num_output_channel, self.num_output_channel, 1, 1)
        # norm = F.conv2d(torch.abs(inputs), weight, beta)

        # 计算余弦相似度 v2
        with torch.no_grad():
            X = inputs
            X = X.view(B, C, -1)
            inner_pro = torch.einsum('oij,ojk->oik', X, X.permute(0, 2, 1))
            norm = torch.norm(X, p=2, dim=2).unsqueeze(0)
            norm_pro = torch.einsum('oij,ojk->oik', norm.permute(0, 2, 1), norm)
            cos_similarities = inner_pro / (norm_pro + 1e-6)
        # 通道拆分运算
        weight = gamma * (1 + cos_similarities)
        weight = weight.view(-1, C, 1, 1)
        beta = beta.repeat(B)
        X = inputs.view(-1, H, W)
        norm = F.conv2d(torch.abs(X), weight, beta, groups=B).view(inputs.size())

        if self.inverse:
            outputs = inputs * norm
        else:
            outputs = inputs / norm
        return outputs


if __name__ == '__main__':
    a = torch.randn(4, 5, 2, 2)
    gdn = GDN(num_output_channel=5)
    b = gdn(a)
    print(a.max())
    print(b.max())
