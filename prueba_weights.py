import torch 
from torch import nn
import numpy as np
from dataclasses import dataclass

@dataclass
class Config:
    in_size = 3
    out_size = 2
    hidden_size = 4

torch.manual_seed(2048)

class Prueba(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.l1 = nn.Linear(config.in_size, config.hidden_size, bias=False)
        self.l2 = nn.Linear(config.hidden_size, config.out_size, bias = False)
        self.relu = nn.ReLU()

    def forward(self, x, targets = None):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x
    

if __name__ == "__main__":
    model = Prueba(Config)
    input_value = torch.tensor([[1, 1, 1]], dtype=torch.float32)
    print(model.l1.weight)
    print(nn.Parameter(model.l1.weight[:3, :]))
    print((model.l1.weight.transpose(0,1) @ model.l2.weight.transpose(0,1)))
    print((model.l1.weight[:3, :].transpose(0,1) @ model.l2.weight[:, :3].transpose(0,1)))
    print(model(input_value))
    model.l1.weight = nn.Parameter(model.l1.weight[:3, :])
    model.l2.weight = nn.Parameter(model.l2.weight[:, :3])
    model.l1.out_features = 3
    model.l2.in_features = 3
    print(model.l1.weight)
    print(model.forward(input_value))
