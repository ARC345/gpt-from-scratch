import torch
import torch.nn as nn
from torch.nn import functional as F

class settings:
    torch_seed = 1337
    
with open('data/tinyshakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()
torch.manual_seed(settings.torch_seed)
