import torch

a = torch.tensor([[[1, 2], [3, 4]], [[1, 2], [3, 5]]])
a = a.view(2, 4)
b, d = a.max(dim=1)
b = b.view(2, 1)
print(a)
print(a.shape)
print(b)
c = abs(a - b)
c = c.view(2, 2, 2)
print(c)