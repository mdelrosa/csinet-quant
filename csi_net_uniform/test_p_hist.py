import torch
import numpy as np

def find(tensor, values):
    return (tensor[...,None] == values).nonzero(as_tuple=True)

bits = 8
n_levels = 2**bits - 1
res = 1/(n_levels)

# verify size of histogram, values
p_hist = torch.zeros(n_levels+1)
p_vals = torch.arange(0.0, 1.0+res, res)
print(f"--- bits: {bits} - res: {res} - n_levels: {n_levels} ---")
print(f"--- p_hist: {p_hist} - p_vals: {p_vals} ---")
print(f"--- p_hist.size(): {p_hist.size()} - p_vals.size(): {p_vals.size()} ---")

N = 128
x = torch.randn((N,N))

# get extrema
x_max = torch.max(x)
x_min = torch.min(x)
x_scaled = (x - x_min) / (x_max - x_min) # scale to [0,1]

x_quant = torch.fake_quantize_per_tensor_affine(x_scaled, res, 0, 0, n_levels) # quantize
x_denorm = x_quant * (x_max - x_min) + x_min # denormalize

vals, counts = x_quant.unique(return_counts=True)

# take batch (vals); update histogram (p_hist)
print(f"p_vals.size(): {p_vals.size()} - vals.size(): {vals.size()}")
for i, val in enumerate(vals):
    temp = torch.isclose(p_vals, torch.ones(p_vals.size(0))*val)
    idx = temp.nonzero().flatten()
    if idx.size(0) > 0:
        j = idx[0]
        p_hist[j] += counts[i]
    else:
        print(f"val={val} not in p_vals")

# print(f"updated p_hist: {p_hist}")
M = torch.sum(p_hist) # total num of elements
entropy = 0
for p in p_hist:
    proba = p / M
    entropy -= proba*torch.log2(proba) if proba > 0 else 0
print(f"entropy: {entropy:4.3f}")

