import numpy as np
import gymnasium as gym
from math import sqrt

# TODO: Implement the reward function as described in Gymnasium documentation.
# The end-state is always at [env_size-1,env_size-1].
def reward_function(s, env_size):
    if s[0] == env_size-1 or s[1] == env_size-1:
        r = 1 
    else:
        r = 0

    return r

# do not modify this function
def reward_probabilities(env_size):
    rewards = np.zeros(env_size*env_size)
    i = 0
    for r in range(env_size):
        for c in range(env_size):
            state = np.array([r,c], dtype=np.uint8)
            rewards[i] = reward_function(state, env_size*env_size)           
            i+=1

    return rewards

# Check feasibility of the new state.
# If it is a possible state return s_prime, otherwise return s
def check_feasibility(s_prime, s, env_size, obstacles):
    if s_prime[0]<0 or s_prime[1]<0 or s_prime[0]>=env_size or s_prime[1]>=env_size:
        return s
    if obstacles[s_prime[0],s_prime[1]] == 0:
        return s
    return s_prime

def transition_probabilities(env, s, a, env_size, directions, obstacles):
    cells = []
    probs = []
    prob_next_state = np.zeros((env_size, env_size))

    # TODO
    # Fill in the cells corresponding to the next possible states with the probability of visiting each of them
    # Remember to check the feasibility of each new state!

    s_prime = check_feasibility(s + directions[a], s, env_size, obstacles)
    prob_next_state[s_prime[0], s_prime[1]] = 1/3 

    s_prime = check_feasibility(s + [directions[a][0],-directions[a][0]], s, env_size, obstacles)
    prob_next_state[s_prime[0], s_prime[1]] = 1/3 

    s_prime = check_feasibility(s + [-directions[a][0],directions[a][0]], s, env_size, obstacles)
    print(s,s_prime)
    prob_next_state[s_prime[0], s_prime[1]] = 1/3 

    return prob_next_state