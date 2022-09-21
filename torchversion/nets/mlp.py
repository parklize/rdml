import torch 
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dims, n_classes):
        
        super(MLP, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dims, 50),
            nn.ReLU(),
            nn.Linear(50, n_classes)
        )
        
    def forward(self, x):
        return self.mlp(x)
        