import random
import numpy as np
# import gymnasium as gym
# import time
# from gymnasium import spaces
# import os
import sklearn
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
import pickle


class VanillaFeatureEncoder:
    def __init__(self, env):
        self.env = env
        
    def encode(self, state):
        return state
    
    @property
    def size(self): 
        return self.env.observation_space.shape[0]

class RBFFeatureEncoder:
    def __init__(self, env): # modify
        self.env = env
        '''
        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        self.rbf_encoder = RBFSampler(gamma=0.999)

        # Scale features to [0, 1] because RBF assume that 
        # all features are centered around 0 and have variance in the same order
        self.scaler = sklearn.preprocessing.StandardScaler()

        # Fit the scaler 
        self.scaler.fit(observation_examples) 
        transformed_observation_examples = self.scaler.transform(observation_examples)

        self.rbf_encoder.fit(observation_examples )


        '''
        self.featureN = 100
        self.n_samples = 10000
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.gamma = 0.9999

        self.sklearn = RBFSampler(gamma = self.gamma, n_components = self.featureN, random_state=1)

        # Sampling a sequence of states to initialize rbf
        sampled_states = np.array([env.observation_space.sample() for _ in range(self.n_samples)])
        self.scaler.fit(sampled_states)
        self.sklearn.fit(self.scaler.transform(sampled_states))

        # TODO init rbf encoder
        
        '''
        self.gridN = 1
        randomness = 1
        self.sigmasq = 0.5
        highv = env.observation_space.high[0]
        lowv = env.observation_space.low[0]
        highposition = env.observation_space.high[1]
        lowposition = env.observation_space.low[1]
        self.grid = []
        for _ in range(0, self.gridN):
            centers = []
            randomx = random.random()/randomness
            randomy = random.random()/randomness
            centers = np.linspace((lowv + randomx, highv + randomy), (lowposition + randomx, highposition + randomy), num = self.featureN )
            #centers = np.linspace((lowv , highv), (lowposition, highposition), num = self.featureN )
            self.grid.append(centers)
        '''
    def encode(self, state): # modify
        # TODO use the rbf encoder to return the features
        '''
        feature = 0
        for grid in self.grid:
            feature += np.exp([( - (np.linalg.norm( state - center ))**2 / (2 * self.sigmasq) ) for center in grid ])
        return feature
        '''
        
        scaled_state = self.scaler.transform([state])
        features = self.sklearn.transform(scaled_state)
        return features.flatten()
        '''
        #transformed_state = self.scaler.transform([state])

        return self.rbf_encoder.transform([state]).flatten()
        '''

    @property
    def size(self): # modify
        # TODO return the number of features
        return 100

class TDLambda_LVFA:
    def __init__(self, env, feature_encoder_cls=RBFFeatureEncoder, alpha=0.01, alpha_decay=1, 
                 gamma=0.9999, epsilon=0.3, epsilon_decay=0.995, final_epsilon=0.2, lambda_=0.9): # modify if you want (e.g. for forward view)
        self.env = env
        self.feature_encoder = feature_encoder_cls(env)
        self.shape = (self.env.action_space.n, self.feature_encoder.size)
        self.weights = np.random.random(self.shape)
        self.traces = np.zeros(self.shape)
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.lambda_ = lambda_
        
    def Q(self, feats):
        feats = feats.reshape(-1,1)
        return self.weights@feats
    
    def update_transition(self, s, action, s_prime, reward, done): # modify
        s_feats = self.feature_encoder.encode(s)
        s_prime_feats = self.feature_encoder.encode(s_prime)
        
        delta = reward + self.gamma * np.max(self.Q(s_prime_feats)) - self.Q(s_feats)[action]
        
        #self.traces *= self.lambda_ * self.gamma 
        self.traces[action] += s_feats
        
        #self.traces[action] = self.gamma*self.lambda_*self.traces[action] + s_feats

        # TODO update the weights
        self.weights[action] += self.alpha * delta * self.traces[action]

        #self.traces[action] = self.gamma*self.lambda_*self.traces[action]
        self.traces *= self.lambda_ * self.gamma 

        if done: self.traces = np.zeros(self.shape)
        
    def update_alpha_epsilon(self): # do not touch
        self.epsilon = max(self.final_epsilon, self.epsilon*self.epsilon_decay)
        self.alpha = self.alpha*self.alpha_decay
        
    def policy(self, state): # do not touch
        state_feats = self.feature_encoder.encode(state)
        return self.Q(state_feats).argmax()
    
    def epsilon_greedy(self, state, epsilon=None): # do not touch
        if epsilon is None: epsilon = self.epsilon
        if random.random()<epsilon:
            return self.env.action_space.sample()
        return self.policy(state)
       
        
    def train(self, n_episodes=200, max_steps_per_episode=200): # do not touch
        print(f'ep | eval | epsilon | alpha')
        for episode in range(n_episodes):
            done = False
            s, _ = self.env.reset()
            self.traces = np.zeros(self.shape)
            for i in range(max_steps_per_episode):
                
                action = self.epsilon_greedy(s)
                s_prime, reward, done, _, _ = self.env.step(action)
                self.update_transition(s, action, s_prime, reward, done)
                
                s = s_prime
                
                if done: break
                
            self.update_alpha_epsilon()

            if episode % 20 == 0:
                print(episode, self.evaluate(), self.epsilon, self.alpha)
                
    def evaluate(self, env=None, n_episodes=10, max_steps_per_episode=200): # do not touch
        if env is None:
            env = self.env
            
        rewards = []
        for episode in range(n_episodes):
            total_reward = 0
            done = False
            s, _ = env.reset()
            for i in range(max_steps_per_episode):
                action = self.policy(s)
                
                s_prime, reward, done, _, _ = env.step(action)
                
                total_reward += reward
                s = s_prime
                if done: break
            
            rewards.append(total_reward)
            
        return np.mean(rewards)

    def save(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, fname):
        return pickle.load(open(fname,'rb'))
