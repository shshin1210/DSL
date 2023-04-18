import torch.nn as nn
import torch.nn.functional as F
import torch

class mlp_depth(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(mlp_depth, self).__init__()
        
        self.input_dim = input_dim
        
        self.fc1 = nn.Linear(input_dim , 512, bias = True) # +2 for added (x,y) dim
        self.fc2 = nn.Linear(512, 512, bias = True)
        self.fc3 = nn.Linear(512, 512, bias = True)
        self.fc4 = nn.Linear(512, 512, bias = True)
        self.fc5 = nn.Linear(512, 512, bias = True)
        self.fc6 = nn.Linear(512, 512, bias = True)
        self.fc7 = nn.Linear(512, 512, bias = True)
        self.fc8 = nn.Linear(512, 128, bias = True)
        self.fc9 = nn.Linear(128, output_dim, bias = True) # 2        
        
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        x = self.fc6(x)
        x = F.relu(x)
        x = self.fc7(x)
        x = F.relu(x)
        x = self.fc8(x)
        x = F.relu(x)
        x = self.fc9(x)
        
        return x
    
class mlp_hyp(nn.Module):
    def __init__(self, input_dim, output_dim, fdim = 1000):
        super(mlp_hyp, self).__init__()
        
        self.input_dim = input_dim
        
        self.fc1 = nn.Linear(input_dim , fdim, bias = True) # +2 for added (x,y) dim
        self.fc2 = nn.Linear(fdim, fdim, bias = True)
        self.fc3 = nn.Linear(fdim, fdim, bias = True)
        self.fc4 = nn.Linear(fdim, fdim, bias = True)
        self.fc5 = nn.Linear(fdim, fdim, bias = True)
        self.fc6 = nn.Linear(fdim, fdim, bias = True)
        self.fc7 = nn.Linear(fdim, fdim, bias = True)
        self.fc8 = nn.Linear(fdim, 128, bias = True)
        self.fc9 = nn.Linear(128, output_dim, bias = True) # 2        
        
    def forward(self, A, b):
        x = torch.cat((A, b), dim=2).flatten()
        x = x.view(-1, self.input_dim)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        x = self.fc6(x)
        x = F.relu(x)
        x = self.fc7(x)
        x = F.relu(x)
        x = self.fc8(x)
        x = F.relu(x)
        x = self.fc9(x)
        
        return x