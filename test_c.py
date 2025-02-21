import torch

def fn(x):
    return x + 1

x = torch.tensor(1)

fn = torch.compile(fn)

print(fn(x))
