import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time

# a = torch.ones((500, 500, 500))
# # print(a)
# start = time.perf_counter()
# a.sum(dim=(0, 1, 2))
# end1 = time.perf_counter()
# a.sum(dim=(2, 1, 0))
# end2 = time.perf_counter()
# print(end1-start, end2-end1)

# rot = torch.fill(torch.zeros((6, 5)), torch.pi/2)
# cos = rot.cos()
# sin = rot.sin()
# rot_matrices = torch.stack([
#     torch.stack([cos, -sin], dim=-1), # (P, T, 2)
#     torch.stack([sin, cos], dim=-1), # (P, T, 2)
#     torch.stack([sin, cos], dim=-1), # (P, T, 2)
# ], dim=-1)
# print(rot_matrices)
threshold = 1e-3
softplus = nn.Softplus(beta=10)
t = torch.linspace(-5, 5, 1000)
softplus = nn.Softplus(beta=20, threshold=1)
blend_t = torch.sigmoid(20 * (t - 0.5))
left_t = softplus(t + threshold)
right_t = softplus(1 + threshold - t)

dist_t = (1 - blend_t) * left_t + blend_t * right_t
plt.plot(t, dist_t)
plt.show()