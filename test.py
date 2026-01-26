import torch
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

a = torch.randn((5, 3))
print(a.any())