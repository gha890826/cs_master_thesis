import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))
a = [1, 2, 3, 4]
print(a)

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

# Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024**3, 1), 'GB')

import torch
import torch.nn as nn
dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
t1 = torch.randn(1, 2)
t2 = torch.randn(1, 2).to(dev)
print(t1)  # tensor([[-0.2678,  1.9252]])
print(t2)  # tensor([[ 0.5117, -3.6247]], device='cuda:0')
t1.to(dev)
print(t1)  # tensor([[-0.2678,  1.9252]])
print(t1.is_cuda)  # False
t1 = t1.to(dev)
print(t1)  # tensor([[-0.2678,  1.9252]], device='cuda:0')
print(t1.is_cuda)  # True


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(1, 2)

    def forward(self, x):
        x = self.l1(x)
        return x


model = M()   # not on cuda
model.to(dev)  # is on cuda (all parameters)
print(next(model.parameters()).is_cuda)  # True
