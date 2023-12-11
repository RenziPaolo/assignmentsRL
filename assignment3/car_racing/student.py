from genericpath import exists
from os import mkdir, remove
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pygad.torchga
import pygad
from torchvision import transforms
import matplotlib.pyplot as plt
#from multiprocessing import Lock

class Policy(nn.Module):
    continuous = True # you can change this

    def __init__(self, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(Policy, self).__init__()
        self.device = device

        self.env = gym.make('CarRacing-v2', continuous=self.continuous, render_mode='human')
        self.env.reset()
        #TODO 
        latent_dim = 100
        hidden_size = 256
        self.VAE = VAE(latent_size=latent_dim).to(self.device)
        self.MDN_RNN = MDN_RNN(input_size = latent_dim + 3, output_size=latent_dim).to(self.device)
        #self.MDN_RNN = MDRNN(latent_dim, 3, hidden_size, 10)
        #self.Clinear = nn.Linear(in_features=latent_dim + hidden_size, out_features=3).to(self.device)
        self.C = LearnableClippingLayer(in_features=latent_dim + hidden_size, out_features=3).to(self.device)
        self.a = torch.tensor([0,0,0]).to(self.device)
        self.OBS_SIZE = 64

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
        #print(state.shape)

        if "numpy" in str(type(state)):
            state = torch.tensor(state)
        if len(state.shape) ==3:
            state = state.unsqueeze(0)
        state = state.permute(0,1,3,2).permute(0,2,1,3)
        mu, logvar = self.VAE.encode(state.float().to(self.device))
        z = self.VAE.latent(mu, logvar)
        
        #print(list(z))
        def my_function():
            if not hasattr(my_function, "is_first_call"):
                my_function.is_first_call = True
            else:
                my_function.is_first_call = False

            if my_function.is_first_call:
                return  self.C(torch.cat((z, torch.zeros((1,256)).to(self.device)),dim=1).to(self.device))
            else:
                return self.a.to(self.device)
        
        self.a = my_function()
        if self.a.dim==2:
            self.a = self.a.unsqueeze(0)
            
        rolloutRNN = torch.concat((self.a, z), dim=1).to(self.device)
        output, (hn, cn) = self.MDN_RNN.forward_lstm(rolloutRNN)

        output = output.squeeze(0)
        hn = hn.squeeze(0)
        
        self.a = self.C(torch.cat((z, hn),dim=1).to(self.device)).to(self.device)
        #print(self.a)
        torch.clip(self.a, min = -1, max = 1 )

        return self.a.cpu().float().squeeze().detach().numpy()

    def train(self):
        # TODO
        #first initialization / skip if load from file
        rollout = []
        rolloutA = []
        rolloutR = []
        num_rolloutVAE = 128*60
        envVAE = gym.make('CarRacing-v2', continuous=False)#, render_mode='human')
        envVAE.reset()
       
        for i in range(num_rolloutVAE):
           
           a = int(np.random.rand() *3 +1)
           observation, reward, terminated, truncated, _ = envVAE.step(a)
           observation = torch.from_numpy(observation/255)
           observation = observation.permute(0,2,1).permute(1,0,2)
           #observation = transform_train(observation)
           rollout.append(observation)
           rolloutA.append(a)
           rolloutR.append(reward)
           if ((i + 1) % (128*5) == 0) or (terminated or truncated):
               state, _ = envVAE.reset()
        
        rollout = torch.stack(rollout, dim=0).to(self.device)
        

        optimizerVAE = torch.optim.Adam(self.VAE.parameters(), lr=2.5e-4)
        schedulerVAE = torch.optim.lr_scheduler.StepLR(optimizerVAE, step_size=10, gamma=0.8)   
        batch_sizeVAE = 128
        num_epochsVAE = 200

        self.trainmodule(self.VAE.to(self.device), optimizerVAE, rollout.float().to(self.device), batch_sizeVAE, num_epochsVAE, schedulerVAE)

        samples = rollout[(np.random.rand(10)*rollout.shape[0]).astype(int)]
        decodedSamples, _, _ = self.VAE.forward(samples.float())

        for index, obs in enumerate(samples):
            plt.subplot(5, 4, 2*index +1)
            obs = torch.movedim(obs, (1, 2, 0), (0, 1, 2))
            plt.imshow(obs.cpu().numpy(), interpolation='nearest')

        for index, dec in enumerate(decodedSamples):
            plt.subplot(5, 4, 2*index +2)
            decoded = torch.movedim(dec, (1, 2, 0), (0, 1, 2))
            plt.imshow(decoded.cpu().detach().numpy(), interpolation="nearest")

        plt.show()

        mu, logvar = self.VAE.encode(rollout.float())
        rolloutZ = self.VAE.latent(mu, logvar).detach().to(self.device)

        rolloutA = torch.tensor(np.array(rolloutA)).to(self.device).detach()

        

        rolloutRNN = torch.concat((rolloutA.detach(), rolloutZ.detach()), dim=1).to(self.device).detach()

        #rolloutH = self.MDN_RNN.forward_lstm(rolloutRNN).to(self.device)

        optimizerRNN = torch.optim.Adam(self.MDN_RNN.parameters(), lr=2.5e-5)
        schedulerRNN = torch.optim.lr_scheduler.StepLR(optimizerRNN, step_size=1000, gamma=1)
        batch_sizeRNN = 32
        num_epochsRNN = 20

        self.trainmodule(self.MDN_RNN.to(self.device), optimizerRNN, rolloutRNN.detach().to(self.device), batch_sizeRNN, num_epochsRNN, schedulerRNN)

        #self.MDN_RNN()

        # Example usage
        
        param = sum(p.numel() for p in self.C.parameters())

        trainer = self.trainGA(param, self.env, self)
        trainer.trainGA(5)
        # print(CMAres)
        # self.C.load_state_dict(CMAres)


        #CMA-ES(C).train
        for _ in range(1):
            rollout = []
            rolloutA = []
            state, _ = self.env.reset()
            for _ in range(128*60):
                state = torch.tensor(np.array(state)).unsqueeze(0)
                
                state = state.to(self.device)
                a = self.act(state)
                observation, reward, terminated, truncated, info = self.env.step(a)
                
                state = torch.from_numpy(observation)
                state = state.permute(0,2,1).permute(1,0,2)
                rollout.append(state/255)
                rolloutA.append(a)
                #if (i + 1) % (32*100) == 0:
                #    state, _ = self.env.reset()
            rollout = torch.stack(rollout, dim=0).to(self.device)
           # rollout = rollout.permute(0,1,3,2).permute(0,2,1,3)

            optimizerVAE = torch.optim.Adam(self.VAE.parameters(), lr=7.5e-5)

            self.trainmodule(self.VAE.to(self.device), optimizerVAE, rollout.float().to(self.device), batch_sizeVAE, num_epochsVAE, schedulerVAE)

            mu, logvar = self.VAE.encode(rollout.float())
            rolloutZ = self.VAE.latent(mu, logvar).detach().to(self.device)

            rolloutA = torch.tensor(np.array(rolloutA)).to(self.device).detach()

            rolloutRNN = torch.concat((rolloutA.detach(), rolloutZ.detach()), dim=1).to(self.device).detach()

            #rolloutH = self.MDN_RNN.forward_lstm(rolloutRNN).to(self.device)

            self.trainmodule(self.MDN_RNN.to(self.device), optimizerRNN, rolloutRNN.detach().to(self.device), batch_sizeRNN, num_epochsRNN, schedulerRNN)

            #MDN_RNN.train()

            # Example usage
            
            trainer.trainGA(10)

        return

    def save(self):
        torch.save(self.state_dict(), 'model.pt')

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
    class trainGA():
        def __init__(self, params, env, model):
            self.env = env
            self.model = model
            self.params = params
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.last_fitness = 0

        def trainGA(self, num_generations):
            #env_lock = Lock()

            def fitness_func(gaclass, solution, sol_idx):
                #with env_lock:
                    env = gym.make('CarRacing-v2', continuous=True)#, render_mode='state_pixels')
                    env.reset()
                    model_weights_dict = pygad.torchga.model_weights_as_dict(model=self.model.C, weights_vector=solution)
                    self.model.C.load_state_dict(model_weights_dict)

                    # play game
                    observation = env.reset()
                    sum_reward = 0
                    terminated = False
                    truncated = False
                    observation = observation[0]
                    while (not (terminated or truncated)) and (sum_reward < 1000):       
                        observation = observation
                        ob_tensor = torch.tensor(observation/255, dtype=torch.float).to(self.device)
                        action = self.model.act(ob_tensor)
                        observation_next, reward, terminated, truncated, _ = env.step(action)
                        observation = observation_next
                        if reward < 0:
                            reward = reward*2
                        sum_reward += reward 
                    print(sum_reward)
                    return sum_reward


            num_generations = num_generations # Number of generations.
            num_parents_mating = 12 # Number of solutions to be selected as parents in the mating pool.

            sol_per_pop = 20 # Number of solutions in the population.
            num_genes = self.params

            
            def on_generation(ga_instance):
                print(f"Generation = {ga_instance.generations_completed}")
                print(f"Fitness    = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]}")
                print(f"Change     = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - self.last_fitness}")
                self.last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]

            def blend_crossover(parents, offspring_size, ga_instance):
                """
                Blend Crossover (Simulated Binary Crossover) for continuous spaces.

                Parameters:
                - parents (numpy.ndarray): Parent solutions.
                - offspring_size (tuple): Size of the offspring.
                - alpha (float): Crossover control parameter.

                Returns:
                - offspring (numpy.ndarray): Offspring solutions.
                """
                alpha=0.25
                offspring = []
                idx = 0

                while len(offspring) < offspring_size[0]:
                    parent1 = parents[idx % parents.shape[0], :].copy()
                    parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

                    rand_values = np.random.uniform(-alpha-0.25, alpha+0.25, size=len(parent1))

                    # Perform blend crossover
                    child1 = 0.5 * ((1 + alpha) * parent1 + (1 - alpha) * parent2 + rand_values)
                    child2 = 0.5 * ((1 + alpha) * parent2 + (1 - alpha) * parent1 + rand_values)

                    offspring.append(child1)
                    offspring.append(child2)

                    idx += 2

                return np.array(offspring)

            def simulated_binary_crossover(parents, offspring_size ,ga_instance):
                """
                Simulated Binary Crossover (SBX) for continuous spaces.

                Parameters:
                parents (numpy.ndarray): Parent solutions.
                offspring_size (tuple): Size of the offspring.
                eta (int): Distribution index for crossover.

                Returns:
                offspring (numpy.ndarray): Offspring solutions.
                """
                eta=5
                offspring = []
                idx = 0

                def normalize_array(arr):
                    min_val = np.min(arr)
                    max_val = np.max(arr)

                    # Avoid division by zero if all elements are the same
                    if min_val == max_val:
                        return np.zeros_like(arr)

                    normalized_arr = (arr - min_val) / (max_val - min_val)
                    return normalized_arr
                # Get parent rankings from ga_instance
                parent_ranks = normalize_array(ga_instance.last_generation_fitness)

                parent_ranks_sorted = np.sort(parent_ranks)[::-1]              

                # Get the number of genes to mutate based on mutation_percent_genes             

                # Convert floats to tuples with a fixed precision (e.g., 2 decimal places)
                element_to_index = {round(value, 5): index for index, value in enumerate(parent_ranks_sorted)}

                # Replace elements in the original list with their corresponding indicesa
                parent_ranks = [element_to_index[round(value, 5)]+1 for value in parent_ranks]

                while len(offspring) < offspring_size[0]:
                    parent1 = parents[idx % parents.shape[0], :].copy()
                    parent2 = parents[(idx + 1) % parents.shape[0], :].copy()
                    parent1rank = parent_ranks[idx % parents.shape[0]]
                    parent2rank = parent_ranks[(idx + 1) % parents.shape[0]]
                    if parent1rank > len(parent_ranks):
                        prob = ga_instance.mutation_percent_genes[0]/100
                    else:
                        prob = ga_instance.mutation_percent_genes[1]/100
                    for i in range(len(parent1)):
                        if np.random.rand() > prob:
                            if abs(parent1[i] - parent2[i]) > 1e-14:  # Avoid division by zero
                                # Intermediate calculations
                                y1 = min(parent1[i], parent2[i])
                                y2 = max(parent1[i], parent2[i])
                                rand = np.random.rand()
                                beta = 1.0 + (2.0 * (y1 - (-1)) / (y2 - y1))
                                alpha = 2.0 - beta**-(eta + 1)
                                
                                if rand <= 1.0 / alpha:
                                    beta_q = (rand * alpha)**(1.0 / (eta + 1))
                                else:
                                    beta_q = (1.0 / (2.0 - rand * alpha))**(1.0 / (eta + 1))

                                # Generating the first child
                                child1 = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))

                                # Intermediate calculations for second child
                                beta = 1.0 + (2.0 * ((1) - y2) / (y2 - y1))
                                alpha = 2.0 - beta**-(eta + 1)

                                if rand <= 1.0 / alpha:
                                    beta_q = (rand * alpha)**(1.0 / (eta + 1))
                                else:
                                    beta_q = (1.0 / (2.0 - rand * alpha))**(1.0 / (eta + 1))

                                # Generating the second child
                                child2 = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))

                                parent1[i] = child1
                                parent2[i] = child2

                    offspring.append(parent1)
                    offspring.append(parent2)

                    idx += 2

                return np.array(offspring)

            
            def custom_crossover(parents, offspring_size, ga_instance):
                """
                Custom Crossover for continuous spaces with mutation based on parent ranking.

                Parameters:
                - parents (numpy.ndarray): Parent solutions.
                - offspring_size (tuple): Size of the offspring.
                - ga_instance (pygad.GA): PyGAD GA instance.

                Returns:
                - offspring (numpy.ndarray): Offspring solutions.
                """
                alpha = 0.1
                offspring = []
                idx = 0

                def normalize_array(arr):
                    min_val = np.min(arr)
                    max_val = np.max(arr)

                    # Avoid division by zero if all elements are the same
                    if min_val == max_val:
                        return np.zeros_like(arr)

                    normalized_arr = (arr - min_val) / (max_val - min_val)
                    return normalized_arr
                # Get parent rankings from ga_instance
                parent_ranks = normalize_array(ga_instance.last_generation_fitness)

                parent_ranks_sorted = np.sort(parent_ranks)[::-1]              

                # Get the number of genes to mutate based on mutation_percent_genes             

                # Convert floats to tuples with a fixed precision (e.g., 2 decimal places)
                element_to_index = {round(value, 5): index for index, value in enumerate(parent_ranks_sorted)}

                # Replace elements in the original list with their corresponding indicesa
                parent_ranks = [element_to_index[round(value, 5)]+1 for value in parent_ranks]

                # # Create a dictionary mapping unique elements to indices
                # element_to_index = {value: index for index, value in enumerate(set(parent_ranks))}

                # # Replace elements in the original list with their corresponding indices
                # parent_ranks = [element_to_index[value]+1 for value in parent_ranks]
                
                #print("ga_instance.last_generation_fitness", ga_instance.last_generation_fitness, "element_to_index", element_to_index, "parent_ranks", parent_ranks)

                while len(offspring) < offspring_size[0]:
                    # Select parents based on their ranks
                    parent1_rank = parent_ranks[idx % parents.shape[0]]
                    parent2_rank = parent_ranks[(idx + 1) % parents.shape[0]]

                    # Adjust mutation strength based on parent rankings
                    mutation_strength = alpha * (parent1_rank+parent2_rank / (len(parent_ranks)*2))
                    parent1 = parents[idx % parents.shape[0], :].copy()
                    parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

                    num_genes_to_mutate1 = 0 
                    num_genes_to_mutate2 = 0 

                    if parent1_rank > len(parent_ranks)/2:
                        num_genes_to_mutate1 += int(np.ceil(ga_instance.mutation_percent_genes[0] * parents.shape[1] / 100))
                    else:
                        num_genes_to_mutate1 += int(np.ceil(ga_instance.mutation_percent_genes[1] * parents.shape[1] / 100))

                    if parent2_rank > len(parent_ranks)/2:
                        num_genes_to_mutate2 += int(np.ceil(ga_instance.mutation_percent_genes[0] * parents.shape[1] / 100))
                    else:
                        num_genes_to_mutate2 += int(np.ceil(ga_instance.mutation_percent_genes[1] * parents.shape[1] / 100))

                    #print("num_genes_to_mutate", num_genes_to_mutate)

                    # Generate random indices for mutation
                    mutation_indices1 = np.random.choice(parent1.shape[0], num_genes_to_mutate1, replace=False)
                    non_mutation_indices1 = complementary_indices = np.setdiff1d(np.arange(len(parent1)), mutation_indices1)
                    mutation_indices2 = np.random.choice(parent2.shape[0], num_genes_to_mutate2, replace=False)
                    non_mutation_indices2 = complementary_indices = np.setdiff1d(np.arange(len(parent2)), mutation_indices2)

                    # Perform blend crossover with mutation at selected indices
                    child1 = torch.zeros(parent1.shape)
                    child2 = torch.zeros(parent2.shape)

                    rand_values1 = np.random.uniform(-mutation_strength, mutation_strength, size=num_genes_to_mutate1)
                    rand_values2 = np.random.uniform(-mutation_strength, mutation_strength, size=num_genes_to_mutate2)

                    parent1nonmutate = parent1.copy()
                    parent2nonmutate = parent2.copy()

                    parent1mutatep = parent1.copy()
                    parent2mutatep = parent2.copy()
                    parent1mutatem = parent1.copy()
                    parent2mutatem = parent2.copy()

                    parent1nonmutate[non_mutation_indices1] = 0
                    parent2nonmutate[non_mutation_indices2] = 0
                    #parent1nonmutate[mutation_indices1] += rand_values1
                    #parent2nonmutate[mutation_indices2] += rand_values2

                    parent1mutatep[mutation_indices1] = (1 + alpha) * parent1[mutation_indices1]
                    parent2mutatem[mutation_indices2] = (1 - alpha) * parent2[mutation_indices2]
                    parent2mutatep[mutation_indices2] = (1 + alpha) * parent2[mutation_indices2]
                    parent1mutatem[mutation_indices1] = (1 - alpha) * parent1[mutation_indices1]

                    parent1_mutated1 = parent1nonmutate + parent1mutatep
                    parent2_mutated1 = parent2nonmutate + parent2mutatem

                    parent2_mutated2 = parent2nonmutate + parent2mutatep
                    parent1_mutated2 = parent1nonmutate + parent1mutatem

                    child1 += 0.5 * ((parent1_mutated1 + parent2_mutated1))
                    child2 += 0.5 * ((parent2_mutated2 + parent1_mutated2))

                    offspring.append(child1)
                    offspring.append(child2)

                    idx += 2

                return np.array(offspring)

            ga_instance = pygad.GA(num_generations=num_generations,
                                num_parents_mating=num_parents_mating,
                                sol_per_pop=sol_per_pop,
                                num_genes=num_genes,
                                fitness_func=fitness_func,
                                on_generation=on_generation,
                                mutation_type="adaptive",
                                keep_elitism=4,
                                crossover_type="scattered",
                                parent_selection_type="rank",
                                mutation_percent_genes = (30,5),
                                parallel_processing=["thread", 6],
                                
                                )

            ga_instance.run()

            # with concurrent.futures.ThreadPoolExecutor() as executor:
            #     future = executor.submit(run_ga_instance, ga_instance)
            #     concurrent.futures.wait([future])

            #ga_instance.plot_fitness()

            # Returning the details of the best solution.
            solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
            print(f"Parameters of the best solution : {solution}")
            print(f"Fitness value of the best solution = {solution_fitness}")
            print(f"Index of the best solution : {solution_idx}")

            if ga_instance.best_solution_generation != -1:
                print(f"Best fitness value reached after {ga_instance.best_solution_generation} generations.")

            # Saving the GA instance.
            model_weights_dict = pygad.torchga.model_weights_as_dict(model=self.model.C, weights_vector=solution)
            self.model.C.load_state_dict(model_weights_dict)
            self.model.save()

            # Loading the saved GA instance.
            #loaded_ga_instance = pygad.load(filename=filename)
            #loaded_ga_instance.plot_fitness()

class LearnableClippingLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(LearnableClippingLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.Tensor([1.0, -1.0, -1.0]))

    def forward(self, x):
        x = self.linear(x)
        x = self.scale * torch.tanh(x)
        return x
    

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
        # Reconstruction loss
        recon_loss = F.mse_loss(out, y, reduction='sum')

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss is the sum of reconstruction loss and KL divergence loss
        total_loss = recon_loss + kl_loss

        return total_loss

    def get_latent_size(self):
        
        return self.latent_size

    def set_device(self, device):
        self.device = device


class VAE1(nn.Module):
    """ Variational Autoencoder (specific for this task)"""

    def __init__(self):
        super().__init__()

        # global variables
        self.CHANNELS = 3
        self.LATENT = 100
        self.OBS_SIZE = 96
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(self.CHANNELS, 16, kernel_size=3, stride=2),  # Convolution with stride 1
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(kernel_size=4, stride=2),  # Average pooling to reduce dimensions
            nn.Flatten(),
            nn.Linear(16*22*22, 2048),  # Adjusted for new flattened size
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 256),
            nn.LeakyReLU(0.2)
        )

        # latent
        self.mean_layer = nn.Linear(256, self.LATENT)
        self.logvar_layer = nn.Linear(256, self.LATENT)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.LATENT, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 16*22*22),
            nn.LeakyReLU(0.2),
            nn.Linear(16*22*22, 3*96*96),
            nn.LeakyReLU(0.2),
        )

        #self.convT = nn.ConvTranspose2d(3*96*96, self.CHANNELS, 96, stride=1)
        self.conv = nn.Conv2d(self.CHANNELS, self.CHANNELS, kernel_size=3, stride=1, padding=1)

    def encode(self, x):
        x = self.encoder(x)
        mu = self.mean_layer(x)
        logsigma =  self.logvar_layer(x)
        return mu, logsigma
    
    def decode(self, x):
        x = self.decoder(x)
        batch_size = 1
        #print(x.shape)
        if len(x.shape)>1:
            batch_size = x.shape[0]
        #print(batch_size)
        x = torch.reshape(x,(batch_size,3,96,96))
        x = self.conv(x)
        x =  F.sigmoid(x)
        #print(x.shape)
        return x

    def latent(self, mu, logsigma):
        eps = torch.randn_like(logsigma).to(self.device)      
        z = eps.mul(logsigma).add_(mu)
        return z

    def forward(self, x):
        mu, logsigma = self.encode(x)
        z = self.latent(mu, logsigma)
        recon_x = self.decode(z)
        return recon_x, mu, logsigma

    def set_device(self, device):
        self.device = device
    
    def save(self, dest):
        if not exists(dest): mkdir(dest)
        else: 
            if exists(dest+self.name.lower()+'.pt'):
                remove(dest+self.name.lower()+'.pt')
        torch.save(self.state_dict(), dest+self.name.lower()+'.pt')

    def load(self, dir): 
        self.load_state_dict(torch.load(dir+self.name.lower()+'.pt', map_location=self.device))

    def loss_function(self, out, y, mu, logvar):
        # Reconstruction loss
        recon_loss = F.mse_loss(out, y, reduction='sum')

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss is the sum of reconstruction loss and KL divergence loss
        total_loss = recon_loss + kl_loss

        return total_loss
    ''''''
    def loss_function1(self, out, y, mu, logvar):
        CE = np.array([F.cross_entropy(out[0], y[0], reduction="sum").detach().cpu() ,F.cross_entropy(out[1], y[1], reduction="sum").detach().cpu(), F.cross_entropy(out[2], y[2], reduction="sum").detach().cpu()]).mean()
        KL = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return CE+KL
    

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
        self.hidden_size = hidden_size
        self.num_mixs = num_mixs
        self.input_size = input_size
        self.output_size = output_size
        self.state = None
        
        self.lstm = nn.LSTM(input_size, hidden_size, 1, batch_first=True)
        self.mdn = MDN(hidden_size, output_size, num_mixs, mdn_units)

    def forward(self, x):
        
        y = None
        x = x.unsqueeze(0) # batch first
        
        if self.state!=None:
            y, state = self.lstm(x, (self.state[0].detach(), self.state[1].detach()))
        else:
            y, state = self.lstm(x)

        #print("y",y,"state", state)
        self.state = state

        pi, sigma, mu = self.mdn(y)
        
        return state, sigma, mu
            
    def forward_lstm(self, x, state=None):
        
        y = None
        x = x.unsqueeze(0) # batch first
        
        if self.state!=None:
            y, state = self.lstm(x, (self.state[0].detach(), self.state[1].detach()))
        else:
            y, state = self.lstm(x)

        #print("y",y,"state", state)
        self.state = state
        
        return y, state
    # outputs, y_batch, mu, logvar
    def loss_function(self, out, y, mu, logvar):
        #BCE = F.binary_cross_entropy(out, y, reduction="sum")
        KL = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return KL

    def get_hidden_size(self):
        return self.hidden_size


    