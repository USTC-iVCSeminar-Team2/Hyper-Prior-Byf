import torch
import torch.nn.functional as F

x = torch.ones((2, 192, 16, 16))
y = -torch.ones((2, 192, 16, 16))
x = x.view((192, 1, 2 * 256))
y = y.view((192, 1, 2 * 256))
I = torch.arange(192).repeat(192)
J = torch.arange(192).repeat((192, 1))
J = torch.einsum('ij->ji', J).reshape(-1)
C = map(lambda i, j: F.cosine_similarity(x[i], y[j]), I, J)
res = torch.stack(list(C), 1).reshape((192,192))
print(res.size())
