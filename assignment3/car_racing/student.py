import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np
from torch import Tensor, List
from torch.utils.data import Dataset
import math
from torch.distributions.normal import Normal
import cma
from queue import Queue
from collections import deque 

class Policy(nn.Module):
    continuous = True # you can change this

    def __init__(self, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(Policy, self).__init__()
        self.device = device

        self.env = gym.make('CarRacing-v2', continuous=self.continuous)#, render_mode='human')
        self.env.reset()
        #TODO 
        latent_dim = 100
        hidden_size = 5
        self.VAE = VAE(latent_size = latent_dim)
        self.MDN_RNN = MDN_RNN(input_size = latent_dim, output_size=latent_dim)
        #self.MDN_RNN = MDRNN(latent_dim, 3, hidden_size, 10)
        self.C = nn.Linear(latent_dim + hidden_size, 3 )
        self.a = [0,0,0]

    def forward(self, x):
        # TODO
        z = self.VAE.encode(x)      
        a = self.C(z, self.MDN_RNN.forward_lstm(z))
        a = torch.clip(a, min = -1, max = 1 )
        h = self.MDN_RNN(z, a, h)

        return a
    
    def act(self, state):
        # TODO
# z = vae.encode(obs)
# a = controller.action([z, h])
# obs, reward, done = env.step(a)
# cumulative_reward += reward
# h = rnn.forward([a, z, h])

        z = self.VAE.encode(state.float())      
        #print(list(z))
        def my_function():
            if not hasattr(my_function, "is_first_call"):
                my_function.is_first_call = True
            else:
                my_function.is_first_call = False

            if my_function.is_first_call:
                self.a = [0,0,0]
            else:
                self.a = self.a
            
        rolloutRNN = torch.concat((self.a, z), dim=1)
        h = self.MDN_RNN.forward_lstm(rolloutRNN)
        self.a = self.C(torch.tensor([z, h]))
        torch.clip(self.a, min = -1, max = 1 )

        return self.a

    def train(self):
        # TODO
        #first initialization / skip if load from file
        rollout = []
        rolloutA = []
        
        num_rolloutVAE = 32*10
       
        for i in range(num_rolloutVAE):
           a = self.env.action_space.sample()
           observation, reward, terminated, truncated, info = self.env.step(a)
           observation = torch.from_numpy(observation/255)
           
           rollout.append(observation)
           rolloutA.append(a)
           #if (i + 1) % (32*100) == 0:
           #    state, _ = self.env.reset()
        
        rollout = torch.stack(rollout, dim=0)
        rollout = rollout.permute(0,1,3,2).permute(0,2,1,3)

        optimizerVAE = torch.optim.Adam(self.VAE.parameters(), lr=1e-4)
        batch_sizeVAE = 32
        num_epochsVAE = 10

        self.trainmodule(self.VAE, optimizerVAE, rollout.float(), batch_sizeVAE, num_epochsVAE)

        mu, logvar = self.VAE.encode(rollout.float())
        rolloutZ = self.VAE.latent(mu, logvar).detach()

        rolloutA = torch.tensor(np.array(rolloutA)).detach()

        rolloutRNN = torch.concat((rolloutA, rolloutZ), dim=1)

        rolloutH = self.MDN_RNN.forward_lstm(rolloutRNN)

        optimizerRNN = torch.optim.Adam(self.MDN_RNN.parameters(), lr=7e-4)
        batch_sizeRNN = 32
        num_epochsRNN = 20

        self.trainmodule(self.MDN_RNN, optimizerRNN, rolloutRNN, batch_sizeRNN, num_epochsRNN)

        #MDN_RNN.train()

        # Example usage
        
        for param in self.C.parameters():
            param
            #print(param)

        cma = cma.CMAEvolutionStrategy(param.float().detach().numpy(), 1)

        #CMA-ES(C).train
        for _ in range(10):
            rollout = []
            state, _ = self.env.reset()
            for _ in range(10000):
                state = torch.tensor(state).permute(0,2,1).permute(1,0,2)
                print(state.shape)
                a = self.act(state)
                observation, reward, terminated, truncated, info = self.env.step(a)
                rollout.append(observation/255)
                observation = state
                
            optimizerVAE = torch.optim.Adam(self.VAE.parameters(), lr=5e-7)
            batch_sizeVAE = 32
            num_epochsVAE = 100

            self.trainmodule(self.VAE, optimizerVAE, rollout, batch_sizeVAE, num_epochsVAE)

            rolloutZ = []
            for _ in range(10000):
                z = self.VAE.encode(rollout)
                observation = self.VAE.encode(z)
                rolloutZ.append(observation)

            optimizerRNN = torch.optim.Adam(self.RNN.parameters(), lr=0.01)
            batch_sizeRNN = 32
            num_epochsRNN = 100

            
            self.trainmodule(self.MDN_RNN, optimizerRNN, rollout, batch_sizeRNN, num_epochsRNN)
        return

    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self):
        self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret

    def trainmodule(self, network, optimizer, data, batch_size,  num_epochs):
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(num_epochs):
            for i in range(0, len(data), batch_size):
                # Get batch
                X_batch = data[i:i+batch_size]
                y_batch = data[i:i+batch_size]


                # Forward pass
                outputs, mu, logvar = network.forward(X_batch)
                loss = network.loss_function(outputs, y_batch, mu, logvar)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Print the loss every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        return

class VAE(nn.Module):
    
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.device = None
        
        # encoder
        self.batch_norm_img = nn.BatchNorm2d(3)
        self.norm_img = nn.BatchNorm1d(3)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.norm1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.norm2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.norm3 = nn.BatchNorm1d(64)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # z
        self.mu = nn.Linear(1024, latent_size)
        self.logvar = nn.Linear(1024, latent_size)
        
        # decoder
        # Assuming the input vector is 1024 elements
        self.fc = nn.Linear(latent_size, 6 * 6 * 256)  # Convert 1024 elements back to 4x4x64 tensor

        # Transposed convolutions

        self.dec_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv4 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.latent(mu, logvar)
        out = self.decode(z)
        
        return out, mu, logvar   
        
    def encode(self, x):
        self.batch_size = 1
        if len(x.shape)>3:
            self.batch_size = x.shape[0]
        
        # if self.batch_size >1: 
        #     x = self.batch_norm_img(x)
        # else:
        #     x = self.norm_img(x)
        out = F.relu(self.conv1(x))
        # if self.batch_size >1: 
        #     out = self.batch_norm1(out)
        # else:
        #     out = self.norm1(out)
        out = F.relu(self.conv2(out))
        # if self.batch_size >1: 
        #     out = self.batch_norm2(out)
        # else:
        #     out = self.norm2(out)
        out = F.relu(self.conv3(out))
        # if self.batch_size >1: 
        #     out = self.batch_norm3(out)
        # else:
        #     out = self.norm3(out)
        out = F.relu(self.adaptive_pool(out))
        out = out.reshape(self.batch_size,1024)
        
        

        mu = self.mu(out)
        logvar = self.logvar(out)
        
        #print("mu", mu, "\n logvar", logvar)

        return mu, logvar
        
    def decode(self, z):
        #batch_size = z.shape[0]
        #print("z",z)
        #print(z.shape)
        out = self.fc(z)
        out = out.view(self.batch_size, 256, 6, 6)
        # out = out.view(-1, 64, 4, 4)  # Reshape to 4x4x64 tensor
        #print("out",out)
        # out = z.view(batch_size, self.latent_size, 1, 1)
        #print(out.shape)

        out = F.relu(self.dec_conv1(out))
        out = F.relu(self.dec_conv2(out))
        out = torch.sigmoid(self.dec_conv3(out))
        out = torch.sigmoid(self.dec_conv4(out))
        
        return out
        
        
    def latent(self, mu, logvar):
        #print("logvar",logvar)
        sigma = torch.exp(0.5*logvar)
        eps = torch.randn_like(logvar).to(self.device)
        z = mu + eps*sigma
        #print("sigma",sigma)
        return z
    
    def obs_to_z(self, x):
        mu, logvar = self.encode(x)
        z = self.latent(mu, logvar)
        
        return z

    def sample(self, z):
        out = self.decode(z)
        
        return out
    
    def loss_function(self, out, y, mu, logvar):
        #print("out",out, "\n y",y)
        CE = F.cross_entropy(out, y)
        KL = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return KL + CE

    def get_latent_size(self):
        
        return self.latent_size

    def set_device(self, device):
        self.device = device

class MDN(nn.Module):
    
    def __init__(self, input_size, output_size, K, units=512):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.K = K
        
        self.l1 = nn.Linear(input_size, 3 * K * output_size)
        
        self.oneDivSqrtTwoPI = 1.0 / np.sqrt(2.0*np.pi)
        
    def forward(self, x):
        batch_size, seq_len = x.shape[0],x.shape[1]

        out = self.l1(x)

        pi, sigma, mu  = torch.split(out, (self.K * self.output_size , self.K * self.output_size, self.K * self.output_size), dim=2)
        
         
        sigma = sigma.view(batch_size, seq_len, self.K, self.output_size)
        sigma = torch.exp(sigma)
        
        mu = mu.view(batch_size, seq_len, self.K, self.output_size)

        pi = pi.view(batch_size, seq_len, self.K, self.output_size)
        pi = F.softmax(pi, dim=2)
        
        return pi, sigma, mu
    
    def gaussian_distribution(self, y, mu, sigma):
        y = y.unsqueeze(2).expand_as(sigma)
        
        out = (y - mu) / sigma
        out = -0.5 * (out * out)
        out = (torch.exp(out) / sigma) * self.oneDivSqrtTwoPI

        return out
    
    def loss(self, y, pi, mu, sigma):

        out = self.gaussian_distribution(y, mu, sigma)
        out = out * pi
        out = torch.sum(out, dim=2)
        
        # kill (inf) nan loss
        out[out <= float(1e-24)] = 1
        
        out = -torch.log(out)
        out = torch.mean(out)
        
        return out

class MDN_RNN(nn.Module):

    def __init__(self, input_size, output_size, mdn_units=512, hidden_size=256, num_mixs=5):
        super(MDN_RNN, self).__init__()
        self.queue = deque(maxlen=5)
        self.hidden_size = hidden_size
        self.num_mixs = num_mixs
        self.input_size = input_size
        self.output_size = output_size
        
        self.lstm = nn.LSTM(input_size, hidden_size, 1, batch_first=True)
        self.mdn = MDN(hidden_size, output_size, num_mixs, mdn_units)

    def forward(self, x):
        
        y = None
        x = x.unsqueeze(0) # batch first
        l = list(self.queue)
        
        if len(l)>0:
            y, state = self.lstm(x, )
        else:
            y, state = self.lstm(x)
        #state = (hn, cn)
        if len(self.queue)>self.queue.maxlen:
            self.queue.pop()
        #print("y",y,"state", state)
        self.queue.append(state)

        pi, sigma, mu = self.mdn(y)
        
        return state, sigma, mu
            
    def forward_lstm(self, x, state=None):
        
        y = None
        x = x.unsqueeze(0) # batch first
        l = list(self.queue)
        
        if len(l)>0:
            y, state = self.lstm(x, )
        else:
            y, state = self.lstm(x)
        #state = (hn, cn)
        if len(self.queue)>self.queue.maxlen:
            self.queue.pop()
        #print("y",y,"state", state)
        self.queue.append(state)
        
        return y, state
# outputs, y_batch, mu, logvar
    def loss_function(self, out, y, mu, logvar):
        #BCE = F.binary_cross_entropy(out, y, reduction="sum")
        KL = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return KL

    def get_hidden_size(self):
        return self.hidden_size


    