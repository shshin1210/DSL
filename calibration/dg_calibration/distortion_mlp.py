import torch
import torch.nn as nn
import torch.nn.functional as F

class distortion_mlp(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(distortion_mlp, self).__init__()
        
        self.input_dim = input_dim
        
        self.fc1 = nn.Linear(input_dim , 512, bias = True)
        self.fc2 = nn.Linear(512, 512, bias = True)
        self.fc3 = nn.Linear(512, 512, bias = True)
        self.fc4 = nn.Linear(512, output_dim, bias = True)
        
    def forward(self, x, y):
        x = torch.cat((x, y), dim=2)
        x = x.view(-1, self.input_dim)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        
        return x