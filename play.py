import torch
import torch.nn.functional as F
import numpy as np

X = torch.randn(2, 5)
T = X.permute(1,0)
A = torch.einsum('ij,jk->jk',X,T)
print(X.t())
print(A)
