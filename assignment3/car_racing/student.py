import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor, List
from gymnasium import env
from torch.utils.data import Dataset
import math

class Policy(nn.Module):
    continuous = True # you can change this

    def __init__(self, device=torch.device('cpu')):
        super(Policy, self).__init__()
        self.device = device
        #TODO 
        latent_dim = 100
        hidden_size = 5
        self.VAE = VAE(in_channels = 96*96*3, latent_dim = latent_dim, hidden_dims = None)
        self.MDN_RNN = MDN_RNN(input_size = 100+3+10, hidden_size = hidden_size, num_layers = 1, dropout = 0.1)
        self.C = nn.linear(latent_dim + hidden_size, ((1,), 1, 1) )

    def forward(self, x):
        # TODO
        z = self.VAE.encode(x[2:4])      
        a = self.C(z, self.MDN_RNN.getHiddenState())
        a = torch.clip(a, min = -1, max = 1 )
        h = self.MDN_RNN(z, a, h)

        return a
    
    def act(self, state):
        # TODO
        z = self.VAE.encode(state[2:4])      
        a = self.C(z, self.MDN_RNN.getHiddenState())
        a = torch.clip(a, min = -1, max = 1 )

        return a

    def train(self):
        # TODO
        #first initialization / skip if load from file
        rollout = []
        for _ in range(10000):
           a = env.ActionSpace.sample()
           observation, reward, terminated, truncated, info = env.step(a)
           rollout.append(observation[2:4])
        

        optimizerVAE = torch.optim.Adam(self.VAE.parameters(), lr=0.01)
        batch_sizeVAE = 32
        num_epochsVAE = 100

        self.trainmodule(self.VAE, optimizerVAE, rollout, batch_sizeVAE, num_epochsVAE)

        rolloutZ = []
        for _ in range(10000):
           z = self.VAE.encode(rollout[2:4])
           observation = self.VAE.encode(z)
           rolloutZ.append(observation)

        optimizerRNN = torch.optim.Adam(self.RNN.parameters(), lr=0.01)
        batch_sizeRNN = 32
        num_epochsRNN = 100

        self.trainmodule(self.MDN_RNN, optimizerRNN, rollout, batch_sizeRNN, num_epochsRNN)
 
        #MDN_RNN.train()

        # Example usage
        
        sigma = 1.0

        cmaes_optimizer = CMAESOptimizer(self.C.parameters(), sigma, )
        xmin_result = cmaes_optimizer.optimize()

        state_dict = self.C.state_dict()
        state_dict = xmin_result
        self.C.load_state_dict(state_dict)

        #print("Optimal solution:", xmin_result)

        #CMA-ES(C).train
        for _ in range(10):
            rollout = []
            state, _ = env.reset()
            for _ in range(10000):
                a = self.act(state)
                observation, reward, terminated, truncated, info = env.step(a)
                rollout.append(observation[2:4])
                observation = state
                

            optimizerVAE = torch.optim.Adam(self.VAE.parameters(), lr=0.01)
            batch_sizeVAE = 32
            num_epochsVAE = 100

            self.trainmodule(self.VAE, optimizerVAE, rollout, batch_sizeVAE, num_epochsVAE)

            rolloutZ = []
            for _ in range(10000):
                z = self.VAE.encode(rollout[2:4])
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

    def trainmodule(network, optimizer, data, batch_size, num_epochs):
        for epoch in range(num_epochs):
            for i in range(0, len(data), batch_size):
                # Get batch
                X_batch = data[i:i+batch_size]
                y_batch = data[i:i+batch_size]


            # Forward pass
            outputs = network(X_batch)
            loss = network.loss_function(outputs, y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print the loss every 100 epochs
            if (epoch + 1) % 100 == 0:
                print(f'RNN: Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        return

class VAE(nn.Module):
    #Variational Auto Encoder Class
    

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


class MDN_RNN(nn.Module):
    #Mixture Density Network Recurrent neural network
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(self).__init__()
        self.RNN = nn.RNN(input_size, hidden_size, num_layers, dropout)
        mix = torch.D.Categorical(torch.ones(5,))
        comp = torch.D.Normal(torch.randn(5,), torch.rand(5,))
        self.gmm = torch.MixtureSameFamily(mix, comp)
        self.RNN_MDN = nn.Sequential(self.RNN, self.gmm) 
        self.input_size = input_size

    def forward(self, h, a, z):
        
        return self.RNN_MDN(np.array([h, a, z]))
    
    def getHiddenState(self):
        input_data = torch.randn((1, 5, self.input_size))
        _, hidden_state = self.RNN(input_data)
        return hidden_state
    
class CustomDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}
        
class CMAESOptimizer:
    def __init__(self, initial_params, sigma, fitness_function, max_evaluations=10000, stop_fitness=1e-10):
        self.N = len(initial_params)
        self.initial_params = torch.tensor(initial_params, dtype=torch.float32, requires_grad=True)
        self.sigma = sigma
        self.fitness_function = fitness_function
        self.max_evaluations = max_evaluations
        self.stop_fitness = stop_fitness

    def _evaluate_population(self, xmeanw, BD):
        lambda_ = len(xmeanw)
        arfitness = torch.zeros(lambda_ + 1)
        arfitness[0] = 2 * abs(self.stop_fitness) + 1

        for k in range(1, lambda_ + 1):
            arz = torch.randn(self.N)
            arx = xmeanw + self.sigma * (BD @ arz)
            arfitness[k] = self.fitness_function(arx)

        return arfitness

    def optimize(self):
        xmeanw = self.initial_params.clone().detach().requires_grad_()
        B = torch.eye(self.N)
        D = torch.eye(self.N)
        BD = B @ D
        C = BD @ BD.t()
        pc = torch.zeros(self.N)
        ps = torch.zeros(self.N)
        cw = torch.ones(self.N) / math.sqrt(self.N)
        chiN = math.sqrt(self.N) * (1 - 1 / (4 * self.N) + 1 / (21 * self.N ** 2))

        count_eval = 0

        while count_eval < self.max_evaluations:
            arfitness = self._evaluate_population(xmeanw, BD)
            if arfitness[0] <= self.stop_fitness:
                break

            # Sort by fitness and compute weighted mean
            _, arindex = arfitness.sort()
            xmeanw = xmeanw[:, arindex[:-1]]

            zmeanw = torch.randn_like(xmeanw)
            xmeanw = xmeanw @ cw
            zmeanw = zmeanw @ cw

            # Adapt covariance matrix
            pc = (1 - 0.25) * pc + math.sqrt(0.25 * (2 - 0.25)) * (BD @ zmeanw)
            C = (1 - 0.25) * C + 0.25 * pc.view(-1, 1) @ pc.view(1, -1)

            # Adapt sigma
            ps = (1 - 0.25) * ps + math.sqrt(0.25 * (2 - 0.25)) * (B @ zmeanw)
            self.sigma = self.sigma * math.exp((ps.norm() - chiN) / chiN / (1 + 0.01 * chiN))

            # Update B and D from C
            if count_eval % (self.N * 10) < 1:
                C = torch.triu(C) + torch.triu(C, 1).t()  # enforce symmetry
                B, D = torch.symeig(C, eigenvectors=True)
                D = torch.diag(torch.sqrt(D))
                BD = B @ D  # for speed up only

            count_eval += 1

            print(f'{count_eval}: {arfitness[0]}')

        xmin = xmeanw[:, arindex[0]]
        return xmin




