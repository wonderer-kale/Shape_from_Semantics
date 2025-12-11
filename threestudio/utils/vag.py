# edit by kw

import torch
import math

def angles_to_vector(elevation, azimuth):
    # degrees → radians
    ele = elevation * math.pi / 180.0
    azi = azimuth * math.pi / 180.0

    x = torch.cos(ele) * torch.cos(azi)
    y = torch.cos(ele) * torch.sin(azi)
    z = torch.sin(ele)
    return torch.stack([x, y, z])

def angle_dot(e1, a1, e2, a2):
    print(f"e1: {e1}, a1: {a1}, e2: {e2}, a2: {a2}")
    v1 = angles_to_vector(e1, a1)
    v2 = angles_to_vector(e2, a2)
    print(f"the shape of v: {v1.shape}, v2: {v2.shape}")
    return torch.dot(v1, v2)

def compute_view_dependent_scale(elevation, azimuth):
    s_list = []
    for e, a in zip(elevation, azimuth):
        device = elevation.device
        w0 = 1 / (1 - angle_dot(e, a, torch.tensor(0.0, device=device), torch.tensor(0.0, device=device))) # front
        w1 = 1 / (1 - angle_dot(e, a, torch.tensor(0.0, device=device), torch.tensor(180.0, device=device)))  # back
        w2 = 1 / (1 - angle_dot(e, a, torch.tensor(0.0, device=device), torch.tensor(90.0, device=device)))  # side
        w3 = 1 / (1 - angle_dot(e, a, torch.tensor(90.0, device=device), torch.tensor(0.0, device=device)))  # overhead

        w_sum = w0 + w1 + w2 + w3
        s0 = 70.0 # hyperparameter to control the overall scale
        W = torch.stack([w0, w1, w2, w3], dim=0)  # shape (4, ...)

        top2 = torch.topk(W, k=2, dim=0).values
        largest_w = top2[0]                
        second_largest_w = top2[1]

        s = s0 * (largest_w - second_largest_w) / (w_sum)
        print(f"s: {s}")
        s_list.append(s)
    s = torch.stack(s_list, dim=0)  # shape (B, )
    return s