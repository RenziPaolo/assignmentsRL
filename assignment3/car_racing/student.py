import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np
from torch import Tensor, List
from torch.utils.data import Dataset
import math

class Policy(nn.Module):
    continuous = True # you can change this

    def __init__(self, device=torch.device('cpu')):
        super(Policy, self).__init__()
        self.device = device

        self.env = gym.make('CarRacing-v2', continuous=self.continuous, render_mode='human')
        self.env.reset()
        #TODO 
        latent_dim = 100
        hidden_size = 5
        self.VAE = VAE(in_channels = 2, latent_dim = latent_dim, hidden_dims = None)
        self.MDN_RNN = MDN_RNN(input_size = 100+3+10, hidden_size = hidden_size, num_layers = 1, dropout = 0.1)
        self.C = nn.Linear(latent_dim + hidden_size, 3 )

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
        for _ in range(1):
           a = self.env.action_space.sample()
           observation, reward, terminated, truncated, info = self.env.step(a)
           rollout.append(observation[2:4])
        

        optimizerVAE = torch.optim.Adam(self.VAE.parameters(), lr=0.01)
        batch_sizeVAE = 32
        num_epochsVAE = 100

        self.trainmodule(self.VAE, optimizerVAE, rollout, batch_sizeVAE, num_epochsVAE)

        
        # for epoch in range(num_epochsVAE):
        #     for i in range(0, len(rollout), batch_sizeVAE):
        #         # Get batch
        #         X_batch = rollout[i:i+batch_sizeVAE]
        #         y_batch = rollout[i:i+batch_sizeVAE]


        #     # Forward pass
        #     outputs = self.VAE(X_batch)
        #     loss = network.loss_function(outputs, y_batch)

        #     # Backward pass and optimization
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()

        #     # Print the loss every 100 epochs
        #     if (epoch + 1) % 100 == 0:
        #         print(f'RNN: Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

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
            state, _ = self.env.reset()
            for _ in range(10000):
                a = self.act(state)
                observation, reward, terminated, truncated, info = self.env.step(a)
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

    def trainmodule(network, modules, optimizer, data, batch_size,  num_epochs):
        print(network)
        for epoch in range(num_epochs):
            for i in range(0, len(data), batch_size):
                # Get batch
                X_batch = torch.tensor(data[i:i+batch_size], dtype=torch.float)
                y_batch = torch.tensor(data[i:i+batch_size], dtype=torch.float)


            # Forward pass
            outputs = network.forward(X_batch)
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
        super(VAE, self).__init__()

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
        super(MDN_RNN, self).__init__()
        self.RNN = nn.RNN(input_size, hidden_size, num_layers, dropout)
        self.gmm = GaussianMixture(10, hidden_size)
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
        Computes the MDN_RNN loss function.
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
    
class GaussianMixture(torch.nn.Module):
    """
    Fits a mixture of k=1,..,K Gaussians to the input data (K is supplied via n_components).
    Input tensors are expected to be flat with dimensions (n: number of samples, d: number of features).
    The model then extends them to (n, 1, d).
    The model parametrization (mu, sigma) is stored as (1, k, d),
    probabilities are shaped (n, k, 1) if they relate to an individual sample,
    or (1, k, 1) if they assign membership probabilities to one of the mixture components.
    """
    def __init__(self, n_components, n_features, covariance_type="full", eps=1.e-6, init_params="kmeans", mu_init=None, var_init=None):
        """
        Initializes the model and brings all tensors into their required shape.
        The class expects data to be fed as a flat tensor in (n, d).
        The class owns:
            x:               torch.Tensor (n, 1, d)
            mu:              torch.Tensor (1, k, d)
            var:             torch.Tensor (1, k, d) or (1, k, d, d)
            pi:              torch.Tensor (1, k, 1)
            covariance_type: str
            eps:             float
            init_params:     str
            log_likelihood:  float
            n_components:    int
            n_features:      int
        args:
            n_components:    int
            n_features:      int
        options:
            mu_init:         torch.Tensor (1, k, d)
            var_init:        torch.Tensor (1, k, d) or (1, k, d, d)
            covariance_type: str
            eps:             float
            init_params:     str
        """
        super(GaussianMixture, self).__init__()

        self.n_components = n_components
        self.n_features = n_features

        self.mu_init = mu_init
        self.var_init = var_init
        self.eps = eps

        self.log_likelihood = -np.inf

        self.covariance_type = covariance_type
        self.init_params = init_params

        assert self.covariance_type in ["full", "diag"]
        assert self.init_params in ["kmeans", "random"]

        self._init_params()


    def _init_params(self):
        if self.mu_init is not None:
            assert self.mu_init.size() == (1, self.n_components, self.n_features), "Input mu_init does not have required tensor dimensions (1, %i, %i)" % (self.n_components, self.n_features)
            # (1, k, d)
            self.mu = torch.nn.Parameter(self.mu_init, requires_grad=False)
        else:
            self.mu = torch.nn.Parameter(torch.randn(1, self.n_components, self.n_features), requires_grad=False)

        if self.covariance_type == "diag":
            if self.var_init is not None:
                # (1, k, d)
                assert self.var_init.size() == (1, self.n_components, self.n_features), "Input var_init does not have required tensor dimensions (1, %i, %i)" % (self.n_components, self.n_features)
                self.var = torch.nn.Parameter(self.var_init, requires_grad=False)
            else:
                self.var = torch.nn.Parameter(torch.ones(1, self.n_components, self.n_features), requires_grad=False)
        elif self.covariance_type == "full":
            if self.var_init is not None:
                # (1, k, d, d)
                assert self.var_init.size() == (1, self.n_components, self.n_features, self.n_features), "Input var_init does not have required tensor dimensions (1, %i, %i, %i)" % (self.n_components, self.n_features, self.n_features)
                self.var = torch.nn.Parameter(self.var_init, requires_grad=False)
            else:
                self.var = torch.nn.Parameter(
                    torch.eye(self.n_features).reshape(1, 1, self.n_features, self.n_features).repeat(1, self.n_components, 1, 1),
                    requires_grad=False
                )

        # (1, k, 1)
        self.pi = torch.nn.Parameter(torch.Tensor(1, self.n_components, 1), requires_grad=False).fill_(1. / self.n_components)
        self.params_fitted = False


    def check_size(self, x):
        if len(x.size()) == 2:
            # (n, d) --> (n, 1, d)
            x = x.unsqueeze(1)

        return x
    
    def fit(self, x, delta=1e-3, n_iter=100, warm_start=False):
        """
        Fits model to the data.
        args:
            x:          torch.Tensor (n, d) or (n, k, d)
        options:
            delta:      float
            n_iter:     int
            warm_start: bool
        """
        if not warm_start and self.params_fitted:
            self._init_params()

        x = self.check_size(x)

        if self.init_params == "kmeans" and self.mu_init is None:
            mu = self.get_kmeans_mu(x, n_centers=self.n_components)
            self.mu.data = mu

        i = 0
        j = np.inf

        while (i <= n_iter) and (j >= delta):

            log_likelihood_old = self.log_likelihood
            mu_old = self.mu
            var_old = self.var

            self.__em(x)
            self.log_likelihood = self.__score(x)

            if torch.isinf(self.log_likelihood.abs()) or torch.isnan(self.log_likelihood):
                device = self.mu.device
                # When the log-likelihood assumes unbound values, reinitialize model
                self.__init__(self.n_components,
                    self.n_features,
                    covariance_type=self.covariance_type,
                    mu_init=self.mu_init,
                    var_init=self.var_init,
                    eps=self.eps)
                for p in self.parameters():
                    p.data = p.data.to(device)
                if self.init_params == "kmeans":
                    self.mu.data, = self.get_kmeans_mu(x, n_centers=self.n_components)

            i += 1
            j = self.log_likelihood - log_likelihood_old

            if j <= delta:
                # When score decreases, revert to old parameters
                self.__update_mu(mu_old)
                self.__update_var(var_old)

        self.params_fitted = True


    def predict(self, x, probs=False):
        """
        Assigns input data to one of the mixture components by evaluating the likelihood under each.
        If probs=True returns normalized probabilities of class membership.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
            probs:      bool
        returns:
            p_k:        torch.Tensor (n, k)
            (or)
            y:          torch.LongTensor (n)
        """
        x = self.check_size(x)

        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)

        if probs:
            p_k = torch.exp(weighted_log_prob)
            return torch.squeeze(p_k / (p_k.sum(1, keepdim=True)))
        else:
            return torch.squeeze(torch.max(weighted_log_prob, 1)[1].type(torch.LongTensor))


    def predict_proba(self, x):
        """
        Returns normalized probabilities of class membership.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
        returns:
            y:          torch.LongTensor (n)
        """
        return self.predict(x, probs=True)


    def sample(self, n):
        """
        Samples from the model.
        args:
            n:          int
        returns:
            x:          torch.Tensor (n, d)
            y:          torch.Tensor (n)
        """
        counts = torch.distributions.multinomial.Multinomial(total_count=n, probs=self.pi.squeeze()).sample()
        x = torch.empty(0, device=counts.device)
        y = torch.cat([torch.full([int(sample)], j, device=counts.device) for j, sample in enumerate(counts)])

        # Only iterate over components with non-zero counts
        for k in np.arange(self.n_components)[counts > 0]: 
            if self.covariance_type == "diag":
                x_k = self.mu[0, k] + torch.randn(int(counts[k]), self.n_features, device=x.device) * torch.sqrt(self.var[0, k])
            elif self.covariance_type == "full":
                d_k = torch.distributions.multivariate_normal.MultivariateNormal(self.mu[0, k], self.var[0, k])
                x_k = torch.stack([d_k.sample() for _ in range(int(counts[k]))])

            x = torch.cat((x, x_k), dim=0)

        return x, y
    
    def __update_mu(self, mu):
        """
        Updates mean to the provided value.
        args:
            mu:         torch.FloatTensor
        """
        assert mu.size() in [(self.n_components, self.n_features), (1, self.n_components, self.n_features)], "Input mu does not have required tensor dimensions (%i, %i) or (1, %i, %i)" % (self.n_components, self.n_features, self.n_components, self.n_features)

        if mu.size() == (self.n_components, self.n_features):
            self.mu = mu.unsqueeze(0)
        elif mu.size() == (1, self.n_components, self.n_features):
            self.mu.data = mu


    def __update_var(self, var):
        """
        Updates variance to the provided value.
        args:
            var:        torch.FloatTensor
        """
        if self.covariance_type == "full":
            assert var.size() in [(self.n_components, self.n_features, self.n_features), (1, self.n_components, self.n_features, self.n_features)], "Input var does not have required tensor dimensions (%i, %i, %i) or (1, %i, %i, %i)" % (self.n_components, self.n_features, self.n_features, self.n_components, self.n_features, self.n_features)

            if var.size() == (self.n_components, self.n_features, self.n_features):
                self.var = var.unsqueeze(0)
            elif var.size() == (1, self.n_components, self.n_features, self.n_features):
                self.var.data = var

        elif self.covariance_type == "diag":
            assert var.size() in [(self.n_components, self.n_features), (1, self.n_components, self.n_features)], "Input var does not have required tensor dimensions (%i, %i) or (1, %i, %i)" % (self.n_components, self.n_features, self.n_components, self.n_features)

            if var.size() == (self.n_components, self.n_features):
                self.var = var.unsqueeze(0)
            elif var.size() == (1, self.n_components, self.n_features):
                self.var.data = var


    def __update_pi(self, pi):
        """
        Updates pi to the provided value.
        args:
            pi:         torch.FloatTensor
        """
        assert pi.size() in [(1, self.n_components, 1)], "Input pi does not have required tensor dimensions (%i, %i, %i)" % (1, self.n_components, 1)

        self.pi.data = pi


        
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




