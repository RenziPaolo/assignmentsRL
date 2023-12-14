import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pygad.torchga
import pygad
from collections import deque as Queue
import matplotlib.pyplot as plt
import cma
from os.path import exists
from os import mkdir, remove
import time
from tqdm import tqdm
#from multiprocessing import Lock

class Policy(nn.Module):
    continuous = False # you can change this

    def __init__(self, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(Policy, self).__init__()
        self.device = device

        self.env = gym.make('CarRacing-v2', continuous=self.continuous, render_mode='human')
        self.env.reset()
        #TODO 
        latent_dim = 32
        hidden_size = 256
        self.vae = VAE(latent_size=latent_dim).to(self.device)
        self.q = Queue(maxlen=2)
        for _ in range(3):
            self.q.append(torch.zeros((32)))
        self.modules_dir = r'C:\Users\Paolo\Downloads\RL\checkpoint'

        self.name = "Nuvolari"

        #self.MDN_RNN = MDN_RNN(input_size = latent_dim + 3, output_size=latent_dim).to(self.device)
        #self.MDN_RNN = MDRNN(latent_dim, 3, hidden_size, 10)
        if self.continuous:
            self.c = c(in_features=(latent_dim), out_features=3).to(self.device)
        else:
            self.c = c(in_features=(latent_dim), out_features=4).to(self.device)
        #self.c = LearnableClippingLayer(in_features=latent_dim, out_features=4).to(self.device)

    def forward(self, x):
        # TODO
        z = self.vae.encode(x)      
        a = self.c(z, self.MDN_RNN.forward_lstm(z))
        a = torch.clip(a, min = -1, max = 1 )
        h = self.MDN_RNN(z, a, h)

        return a
    
    def act(self, state):
        # TODO

        if "numpy" in str(type(state)):
            state = torch.tensor(state)
        if len(state.shape) ==3:
            state = state.unsqueeze(0)
        if state.shape[1] == 96:
            state = state.permute(0,1,3,2).permute(0,2,1,3).to(self.device)
        mu, logvar = self.vae.encode(state.float())
        z = self.vae.latent(mu, logvar)
        
        # l = []
        # while len(self.q) > 0:
        #     l.append(self.q.pop().flatten().to(self.device))
        # for elem in l:
        #     self.q.append(elem)
        # zold = torch.stack(l, dim=0).to(self.device)
        #zold.permute(1,0,2)
        #zold = zold.squeeze(0)
        input = z
        #input = torch.cat((z,zold),dim=0).flatten()
        self.a = self.c(input)
        #self.q.pop()
        #self.q.append(z)
        self.a = self.a.cpu().detach()
        if self.continuous:

            # Check for NaN values
            nan_mask = torch.isnan(self.a)

            # Substitute NaN values with 0
            self.a = torch.where(nan_mask, torch.tensor(0.0), self.a)
            


            self.a = torch.clip(self.a, min = -1.0, max = 1.0 ).numpy()
            if self.a.shape[0] == 1: self.a = self.a[0]

            self.a[1] = abs(self.a[1])
            self.a[2] = abs(self.a[2])
            ret = self.a
        else:
            ret = int(torch.argmax(self.a))+1

        

        return ret
    
# ctallec function:
    def unflatten_parameters(self, params, example, device):
        """ Unflatten parameters. Note: example is generator of parameters (module.parameters()), used to reshape params """
        
        params = torch.Tensor(params).to(device)
        idx = 0
        unflattened = []
        for e_p in example:
            unflattened += [params[idx:idx + e_p.numel()].view(e_p.size())]
            idx += e_p.numel()
        return unflattened


    def evaluate(self, solutions, results, render=False, num_eval=12):
        print("Evaluating...")
        index_min = np.argmin(results)
        best_guess = solutions[index_min]
        restimates = []

        p_list = []
        for s_id in range(num_eval):
            p_list.append((s_id, best_guess))

        for _ in tqdm(range(num_eval)):
            value = self.rollout(self, best_guess, device=self.device, render=render)
            restimates.append(value)
        
        return best_guess, np.mean(restimates)

    def rollout(self, agent, params=None, limit=1000, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), render=False):
        """ Execute a rollout and returns minus cumulative reward. """

        render_mode = 'human' if render else 'rgb_array'
        self.env = gym.make('CarRacing-v2', continuous=self.continuous, render_mode=render_mode)

        if params is not None:
            params = self.unflatten_parameters(params, agent.c.parameters(), device)

            # load parameters into agent controller
            for p, p_0 in zip(agent.c.parameters(), params):
                p.data.copy_(p_0)

        # ####DEGUB####
        # for p in agent.c.parameters():
        #     print('new parameters: {}'.format(p))
        #     break
        # ####DEGUB####

        obs, _ = self.env.reset()
        cumulative = 0
        done = False

        for _ in range(limit):
            action = self.act(obs) 
            obs, reward, terminated, truncated, _ = self.env.step(action)
                
            done = terminated 
            if done: break

            cumulative += reward # 50 100 -50

        # print("cumulative: {}".format(cumulative))
        # reward "temperature"
        return (1000 - cumulative) # 950 900 1050

    def train(self):
        # TODO
        #first initialization / skip if load from file

    

        rolloutVAE = []
        #rolloutA = []
        rolloutR = []
        num_rolloutVAE = 128*50
        envVAE = gym.make('CarRacing-v2', continuous=self.continuous)#, render_mode='human')
        state, _ = envVAE.reset()
        train_vae = False
        if train_vae == False:
            num_rolloutVAE = 1

        for i in range(num_rolloutVAE):
            if self.continuous:
                a = envVAE.action_space.sample()
            else:
                a = self.act(state)

            observation, reward, terminated, truncated, _ = envVAE.step(a)
            observation = torch.from_numpy(observation/255)
            observation = observation.permute(0,2,1).permute(1,0,2)
            #observation = transform_train(observation)
            rolloutVAE.append(observation)
            #rolloutA.append(a)
            rolloutR.append(reward)
            if terminated or truncated:
                state, _ = envVAE.reset()
        
        rolloutVAE = torch.stack(rolloutVAE, dim=0).float().to(self.device)
        

        optimizerVAE1 = torch.optim.Adam(self.vae.parameters(), lr=1e-3)
        optimizerVAE2 = torch.optim.Adam(self.vae.parameters(), lr=1e-4)
        optimizerVAE3 = torch.optim.Adam(self.vae.parameters(), lr=1e-6)
        optimizerVAE4 = torch.optim.Adam(self.vae.parameters(), lr=1e-7)
        optimizerVAE4 = torch.optim.Adam(self.vae.parameters(), lr=1e-8)
        schedulerVAE = torch.optim.lr_scheduler.StepLR(optimizerVAE1, step_size=10, gamma=0.8)   
        batch_sizeVAE = 128
        num_epochsVAE = 100
        #self.vae = torch.load('vae2.pt')
        if train_vae == True:
            #self.vae = torch.load('vae.pt').to(self.device)
            print("train 1")
            self.trainmodule(self.vae, optimizerVAE1, rolloutVAE, batch_sizeVAE, 50, schedulerVAE)
            self.trainmodule(self.vae, optimizerVAE2, rolloutVAE, batch_sizeVAE, 100, schedulerVAE)
            self.trainmodule(self.vae, optimizerVAE3, rolloutVAE, batch_sizeVAE, 100, schedulerVAE)
            self.trainmodule(self.vae, optimizerVAE4, rolloutVAE, batch_sizeVAE, 100, schedulerVAE)
            self.trainmodule(self.vae, optimizerVAE4, rolloutVAE, batch_sizeVAE, 100, schedulerVAE)
            #save vae


           
        
            samples = rolloutVAE[(np.random.rand(10)*rolloutVAE.shape[0]).astype(int)]
            decodedSamples, _, _ = self.vae.forward(samples.float())
            torch.save(self.vae, 'vae1.pt')
            for index, obs in enumerate(samples):
                plt.subplot(5, 4, 2*index +1)
                obs = torch.movedim(obs, (1, 2, 0), (0, 1, 2))
                plt.imshow(obs.cpu().numpy(), interpolation='nearest')

            for index, dec in enumerate(decodedSamples):
                plt.subplot(5, 4, 2*index +2)
                decoded = torch.movedim(dec, (1, 2, 0), (0, 1, 2))
                plt.imshow(decoded.cpu().detach().numpy(), interpolation="nearest")

            plt.show()
            return
        else:
            #load vae 
            self.vae = torch.load('vae1.pt').to(self.device)

        ###DEGUB####
        # for p in self.c.parameters():
        #     print('previous parameters: {}'.format(p))
        #     break
        ###DEGUB####

        self.cma_train()
        return
    
    def savedir(self, dest):
        if not exists(dest): mkdir(dest)
        else: 
            if exists(dest+self.name.lower()+'.pt'):
                remove(dest+self.name.lower()+'.pt')
        torch.save(self.state_dict(), dest+self.name.lower()+'.pt')

    def save(self):
        torch.save(self.state_dict(), 'model1.pt')

    def load(self):
        self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret

    def trainmodule(self, network, optimizer, data, batch_size,  num_epochs, scheduler):
        #torch.autograd.set_detect_anomaly(True)
        #network.train()
        for epoch in range(num_epochs):
            for i in range(0, len(data), batch_size):
                # Get batch
                X_batch = data[i:i+batch_size]
                y_batch = data[i:i+batch_size]


                # Forward pass
                outputs, mu, logvar = network.forward(X_batch)
                loss = network.loss_function(outputs, y_batch, mu, logvar)

                # Backward pass and optimization
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                #scheduler.step()

            # Print the loss every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        return
    
    def cma_train(self):
        # training parameters
        pop_size = 10
        n_samples = 2 
        generation = 0
        target_return = 400

        # log variables
        log_step = 3 # print log each n steps
        display = True
        render = False
        
        # define current best and load parameters
        cur_best = 100000000000 # max cap

        print("Attempting to load previous best...")
        if exists(self.modules_dir+'controller.pt') and exists(self.modules_dir+'cur_best.bk'):
            # state = torch.load(c_checkpoint, map_location=self.device)
            with open(self.modules_dir+'cur_best.bk', 'r') as f : 
                for i in f : cur_best = float(i)
            self.c = self.c.load(self.modules_dir)
            print("Previous best was {}...".format(-cur_best))

        ####DEGUB####
        for p in self.c.parameters():
            print('previous parameters: {}'.format(p))
            break
        ####DEGUB####

        params = self.c.parameters()
        flat_params = torch.cat([p.detach().view(-1) for p in params], dim=0).cpu().numpy()
        es = cma.CMAEvolutionStrategy(
            flat_params, 
            0.2, 
            {'popsize':pop_size}
        )

        print("Starting CMA training")
        print("Generation {}".format(generation+1))
        start_time = time.time()

        while not es.stop(): # and generation < 20:

            if cur_best is not None and - cur_best > target_return:
                print("Already better than target, breaking...")
                break

            # compute solutions
            r_list = [0] * pop_size  # result list
            solutions = es.ask()

            if display: pbar = tqdm(total=pop_size * n_samples)
            
            for s_id, params in enumerate(solutions):
                for _ in range(n_samples):
                    r_list[s_id] += self.rollout(self, params, device=self.device) / n_samples
                    if display: pbar.update(1)

            if display: pbar.close()

            es.tell(solutions, r_list)
            es.disp()

            # evaluation and saving
            if  generation % log_step == log_step - 1: render = True
            best_params, best = self.evaluate(solutions, r_list)
            print("Current evaluation: {}".format(best)) # -950 -900 -1050

            if not cur_best or cur_best > best: # 950 900 900
                cur_best = best

                print("Saving new best with value {}...".format(cur_best))
                # print("NEW Plus Saving new best with value {}...".format(cur_best))
    
                # load parameters into controller
                unflat_best_params = self.unflatten_parameters(best_params, self.c.parameters(), self.device)
                for p, p_0 in zip(self.c.parameters(), unflat_best_params):
                    p.data.copy_(p_0)
                self.savedir(self.modules_dir)
                with open(self.modules_dir+'cur_best.bk', 'w') as f: f.write(str(cur_best))
                self.save()
                self.evaluate(solutions, r_list, render=True, num_eval=3)

            if best <= target_return: #target_return:
                # print("cur best {}".format(- cur_best))
                print("Terminating controller training with value {}...".format(cur_best))
                break

            generation += 1
            print("Generation {}".format(generation+1))
            
        return

class c(nn.Module):
    def __init__(self, in_features, out_features):
        super(c, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        

    def forward(self, x):
        x = self.fc(x)
        #x = torch.sigmoid(x)
        return x
    
class c1(nn.Module):
    def __init__(self, in_features, out_features):
        super(c1, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.fc(x)
    

class VAE(nn.Module):
    
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.device = None
        
        # encoder
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
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
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.adaptive_pool(out))
        out = out.reshape(self.batch_size,1024)

        mu = self.mu(out)
        logvar = self.logvar(out)

        return mu, logvar
        
    def decode(self, z):
        out = self.fc(z)
        out = out.view(self.batch_size, 256, 6, 6)

        out = F.relu(self.dec_conv1(out))
        out = F.relu(self.dec_conv2(out))
        out = F.relu(self.dec_conv3(out))
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
        # Reconstruction loss
        recon_loss = F.mse_loss(out, y, reduction='sum')

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss is the sum of reconstruction loss and KL divergence loss
        total_loss = recon_loss + kl_loss

        return total_loss
    
    def loss_function1(self, x, x_hat, mean, log_var):

        bce_loss = nn.BCEWithLogitsLoss(reduction='sum')
        reproduction_loss = bce_loss(x_hat, x)  
        KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

        return reproduction_loss + KLD

    def get_latent_size(self):
        
        return self.latent_size

    def set_device(self, device):
        self.device = device