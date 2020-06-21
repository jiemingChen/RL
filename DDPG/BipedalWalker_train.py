import gym
import random
import torch
import numpy as np
from collections import deque

from agent import Agent
from algorithm import DDPG
from model import Model 

from pdb import set_trace
from env import ContinuousCartPoleEnv
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
NOISE = 0.05

def train(n_episodes=5000, max_t=700):
    env = gym.make('BipedalWalker-v2')
    env.seed(10)

    scores_deque = deque(maxlen=100)
    scores = []
    max_score = -np.Inf

    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    model = Model(state_size=obs_size, action_size=action_size)
    target_model = Model(state_size=obs_size, action_size=action_size)

    alg =  DDPG(model,target_model, gamma=0.99, tau=1e-3, actor_lr=1e-4, critic_lr=3e-4)
    agent = Agent(alg, BUFFER_SIZE, BATCH_SIZE, seed=10)

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        #agent.reset()
        score = 0

        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward*0.01, next_state, done)
            state = next_state
            score += reward
            if done:
                break 

        scores_deque.append(score)
        scores.append(score)
        #print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_deque), score), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))   
        if np.mean(scores_deque) > 200:
            torch.save(agent.alg.model.actor_model.state_dict(), 'walker_actor.pth')
            torch.save(agent.alg.model.critic_model.state_dict(), 'walker_critic.pth')
            break

    return scores


train()