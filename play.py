import torch
import torch.nn.functional as F
import numpy as np

X = torch.ones(8, 192, 16, 16)
X = X.view(8, 192, -1)
A = torch.einsum('oij,ojk->oik', X, X.permute(0, 2, 1))
norm = torch.norm(X, p=2, dim=2).unsqueeze(0)
B = torch.einsum('oij,ojk->oik', norm.permute(0, 2, 1), norm)
cos_similarities = A / (B + 1e-6)

gamma = torch.randn(192, 192)
weight = gamma * cos_similarities
weight = weight.view(-1, 192, 1, 1)
beta = torch.randn(192).repeat(8)
X = X.view(-1, 16, 16)
norm = F.conv2d(torch.abs(X), weight, beta, groups=8)
norm = norm.view(8, 192, 16, 16)
print(norm.size())
