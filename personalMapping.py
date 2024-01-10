import random
import torch
from torch import  Tensor, nn
from typing import List
import torch.nn.functional as F
import math

class AttentionNet(nn.Module):

    def __init__(self, input_dim:int, hidden_states:List[int] ):
        super().__init__()
        self.dim = input_dim

        self.Q = nn.Linear(self.dim, self.dim)
        self.K = nn.Linear(self.dim, self.dim)
        self.V = nn.Linear(self.dim, self.dim)

        self.fc = []
        self.fc.append(nn.Linear(self.dim, hidden_states[0]))
        self.fc.append(nn.ReLU(inplace=False))
        self.fc.append(nn.Linear(hidden_states[0], hidden_states[1]))
        self.fc.append(nn.ReLU(inplace=False))
        self.fc.append(nn.Linear(hidden_states[1], hidden_states[2]))
        # for dim in hidden_states:
        #     self.fc.append(nn.Linear(self.dim, dim))
        #     self.fc.append(nn.ReLU(inplace=False))
        #     self.dim = dim
        self.mlp = nn.Sequential(*self.fc)

    def forward(self,x:torch.Tensor):
        
        query = self.Q(x.clone())
        key = self.K(x.clone())
        value = self.V(x.clone())
        atten_scores = F.softmax(torch.matmul(query, key.transpose(-2,-1))/math.sqrt(query.shape[-1]), dim=-1).clone()
        value_new = torch.matmul(atten_scores, value)
        out = self.mlp(value_new)
        
        return out

class MetaNet(nn.Module):

    def __init__(self, input_dim:int,  layers_dim:List[int]):
        super().__init__()
        self.input_dim = input_dim
        self.fc = []
        for dim in layers_dim:
            self.fc.append(nn.Linear(self.input_dim, dim))
            self.input_dim = dim
        self.mlp = nn.Sequential(*self.fc)
    def forward(self, x:torch.Tensor):

        out = self.mlp(x)
        return out
    

    


        
