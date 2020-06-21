import torch
import torch.nn as nn
import torch.nn.functional as F 

class Model(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=20):
        super(Model, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, action_size)

    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = F.softmax(self.fc2(x), dim=1)
        return x
