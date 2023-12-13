import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pygad.torchga
import pygad
from queue import LifoQueue as Queue
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

        self.modules_dir = r'C:\Users\Paolo\Downloads\RL\checkpoint'

        self.name = "Nuvolari"

        #self.MDN_RNN = MDN_RNN(input_size = latent_dim + 3, output_size=latent_dim).to(self.device)
        #self.MDN_RNN = MDRNN(latent_dim, 3, hidden_size, 10)
        self.c = c(in_features=latent_dim, out_features=4).to(self.device)
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

        if not ('q' in locals()):
            q = Queue(maxsize=3)
            for _ in range(3):
                q.put(torch.zeros((3)))

        if "numpy" in str(type(state)):
            state = torch.tensor(state)
        if len(state.shape) ==3:
            state = state.unsqueeze(0)
        if state.shape[1] == 96:
            state = state.permute(0,1,3,2).permute(0,2,1,3).to(self.device)
        mu, logvar = self.vae.encode(state.float())
        z = self.vae.latent(mu, logvar)
        q.get()
        q.put(z)
        l = []
        while not(q.empty):
            l.append(q.get())
        for i in range(len(l)):
            q.put(l[i])

        self.a = self.c(z)
        torch.clip(self.a, min = 0, max = 5 )
        self.a = self.a.cpu().detach()

        #print(self.a)

        return int(torch.argmax(self.a))
    
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
        num_rolloutVAE = 128*120
        envVAE = gym.make('CarRacing-v2', continuous=self.continuous)#, render_mode='human')
        envVAE.reset()
        train_vae = False
        if train_vae == False:
            num_rolloutVAE = 1

        for i in range(num_rolloutVAE):
           
           a = int(np.random.rand() *3 +1)
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
        optimizerVAE2 = torch.optim.Adam(self.vae.parameters(), lr=7.5e-4)
        optimizerVAE3 = torch.optim.Adam(self.vae.parameters(), lr=5e-6)
        optimizerVAE4 = torch.optim.Adam(self.vae.parameters(), lr=1e-7)
        optimizerVAE4 = torch.optim.Adam(self.vae.parameters(), lr=1e-8)
        schedulerVAE = torch.optim.lr_scheduler.StepLR(optimizerVAE1, step_size=10, gamma=0.8)   
        batch_sizeVAE = 128
        num_epochsVAE = 100
        #self.vae = torch.load('vae2.pt')
        if train_vae == True:
            print("train 1")
            self.trainmodule(self.vae, optimizerVAE1, rolloutVAE, batch_sizeVAE, 70, schedulerVAE)
            self.trainmodule(self.vae, optimizerVAE2, rolloutVAE, batch_sizeVAE, 100, schedulerVAE)
            self.trainmodule(self.vae, optimizerVAE3, rolloutVAE, batch_sizeVAE, 100, schedulerVAE)
            self.trainmodule(self.vae, optimizerVAE4, rolloutVAE, batch_sizeVAE, 100, schedulerVAE)
            self.trainmodule(self.vae, optimizerVAE4, rolloutVAE, batch_sizeVAE, 100, schedulerVAE)
            #save vae


           
        
            samples = rolloutVAE[(np.random.rand(10)*rolloutVAE.shape[0]).astype(int)]
            decodedSamples, _, _ = self.vae.forward(samples.float())
            torch.save(self.vae, 'vae.pt')
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
            self.vae = torch.load('vae.pt').to(self.device)

        #self.trainmodule(self.vae, optimizerVAE, rollout, batch_sizeVAE, num_epochsVAE, schedulerVAE)

        #mu, logvar = self.vae.encode(rollout.float())
        #rolloutZ = self.vae.latent(mu, logvar).detach().to(self.device)

        #rolloutA = torch.tensor(np.array(rolloutA)).to(self.device).detach()

        #rolloutRNN = torch.concat((rolloutA.detach(), rolloutZ.detach()), dim=1).to(self.device).detach()

        #rolloutH = self.MDN_RNN.forward_lstm(rolloutRNN).to(self.device)

        # optimizerRNN = torch.optim.Adam(self.MDN_RNN.parameters(), lr=2.5e-5)
        # schedulerRNN = torch.optim.lr_scheduler.StepLR(optimizerRNN, step_size=1000, gamma=1)
        # batch_sizeRNN = 32
        # num_epochsRNN = 20

        #self.trainmodule(self.MDN_RNN.to(self.device), optimizerRNN, rolloutRNN.detach().to(self.device), batch_sizeRNN, num_epochsRNN, schedulerRNN)

        ###DEGUB####
        # for p in self.c.parameters():
        #     print('previous parameters: {}'.format(p))
        #     break
        ###DEGUB####

        # training parameters
        pop_size = 6
        n_samples = 2 
        generation = 0
        target_return = 950 + 1000

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

            if - best > target_return: #target_return:
                # print("cur best {}".format(- cur_best))
                print("Terminating controller training with value {}...".format(cur_best))
                break

            generation += 1
            print("Generation {}".format(generation+1))
            
        return


        
        #param = sum(p.numel() for p in self.c.parameters())

        #trainer = self.trainGA(param, self.env, self)
        #print("train C")
        #trainer.trainGA(10)

        envVAE = gym.make('CarRacing-v2', continuous=self.continuous)#, render_mode='human')
        for _ in range(1):
            rollout = []
            #rolloutA = []
            state, _ = envVAE.reset()
            for _ in range(128*60):
                state = torch.tensor(np.array(state)).unsqueeze(0)
                
                state = state.to(self.device)
                a = self.act(state)
                observation, reward, terminated, truncated, info = envVAE.step(a)
                
                state = torch.from_numpy(observation)
                state = state.permute(0,2,1).permute(1,0,2)
                rollout.append(state/255)
                #rolloutA.append(a)
                if ((i + 1) % (128*5) == 0) or (terminated or truncated):
                    state, _ = envVAE.reset()
            rollout = torch.stack(rollout, dim=0).to(self.device)
           #rollout = rollout.permute(0,1,3,2).permute(0,2,1,3)

        
        optimizerVAE1 = torch.optim.Adam(self.vae.parameters(), lr=5e-5)
        optimizerVAE2 = torch.optim.Adam(self.vae.parameters(), lr=1e-5)
        optimizerVAE3 = torch.optim.Adam(self.vae.parameters(), lr=7.5e-6)

        if train_vae == True:
            print("train 1")
            self.trainmodule(self.vae, optimizerVAE1, rollout, batch_sizeVAE, num_epochsVAE, schedulerVAE)
            self.trainmodule(self.vae, optimizerVAE2, rollout, batch_sizeVAE, num_epochsVAE, schedulerVAE)
            self.trainmodule(self.vae, optimizerVAE3, rollout, batch_sizeVAE, num_epochsVAE, schedulerVAE)
            #save vae

            samples = rollout[(np.random.rand(10)*rollout.shape[0]).astype(int)]
            decodedSamples, _, _ = self.vae.forward(samples.float())
            torch.save(self.vae, 'vae2.pt')
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
            self.vae = torch.load('vae.pt')

            rollout = None

            #mu, logvar = self.vae.encode(rollout.float())
            #rolloutZ = self.vae.latent(mu, logvar).detach().to(self.device)

            #rolloutA = torch.tensor(np.array(rolloutA)).to(self.device).detach()

            #rolloutRNN = torch.concat((rolloutA.detach(), rolloutZ.detach()), dim=1).to(self.device).detach()

            #rolloutH = self.MDN_RNN.forward_lstm(rolloutRNN).to(self.device)

            #self.trainmodule(self.MDN_RNN.to(self.device), optimizerRNN, rolloutRNN.detach().to(self.device), batch_sizeRNN, num_epochsRNN, schedulerRNN)
            trainer.trainGA(20)

        return
    def savedir(self, dest):
        if not exists(dest): mkdir(dest)
        else: 
            if exists(dest+self.name.lower()+'.pt'):
                remove(dest+self.name.lower()+'.pt')
        torch.save(self.state_dict(), dest+self.name.lower()+'.pt')

    def save(self):
        torch.save(self.state_dict(), 'model3.pt')

    def load(self):
        self.load_state_dict(torch.load('model1316.pt', map_location=self.device))

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
                    env = gym.make('CarRacing-v2', continuous=False)#, render_mode='state_pixels')
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
                    print(sum_reward*2)
                    return sum_reward*2
            
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
            
            num_generations = num_generations # Number of generations.
            num_parents_mating = 4 # Number of solutions to be selected as parents in the mating pool.

            sol_per_pop = 6 # Number of solutions in the population.
            num_genes = self.params

            ga_instance = pygad.GA(num_generations=num_generations,
                                num_parents_mating=num_parents_mating,
                                sol_per_pop=sol_per_pop,
                                num_genes=num_genes,
                                fitness_func=fitness_func,
                                on_generation=on_generation,
                                mutation_type="adaptive",
                                keep_elitism=2,
                                crossover_type=custom_crossover,
                                parent_selection_type="rws",
                                mutation_percent_genes = (10,2),
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