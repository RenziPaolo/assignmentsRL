import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Policy(nn.Module):
    continuous = False # you can change this

    def __init__(self, device=torch.device('cpu')):
        super(Policy, self).__init__()
        self.device = device
        #TODO 
        #self.VAE
        #self.MDN-RNN
        #self.C

    def forward(self, x):
        # TODO
        #z, _ = V(x)
        #h = MDN-RNN(z, a, h)
        #a = C(z,h)

        return x
    
    def act(self, state):
        # TODO
        #z,_ = V(state)
        #h = MDN-RNN(z, a, h)
        #a = C(z,h)

        return 

    def train(self):
        # TODO
        #VAE.train()
        #MDN-RNN.train()
        #CMA-ES(C)


        return

    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self):
        self.load_state_dict(torch.load('model.pt'), map_location=self.device)

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
