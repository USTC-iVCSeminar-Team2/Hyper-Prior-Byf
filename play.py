"""
for doing experiment
"""
import torch
import numpy as np

x = torch.tensor(-3.,requires_grad=True)
y = x
y.backward()
print(x.grad)