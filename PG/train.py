import gym
from collections import deque
import torch
import numpy as np

from agent import Agent
from model import Model
from algorithm import PolicyGradient


LR = 2e-2              # learning rate 
PRINT_EVERY = 100
SHOW_EVERY = 100
LOAD_MODEL = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(n_episodes=5000, max_t=1000,  gamma=1.0):
    env = gym.make('CartPole-v0')
    env.seed(0)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n 

    model = Model(state_dim, action_dim, seed=1)
    alg = PolicyGradient(model, device=device, lr=LR)
    agent = Agent(alg, state_dim, action_dim, device)
    if LOAD_MODEL:
        agent.alg.model.load_state_dict(torch.load('checkpoint2.pth'))

    scores_deque = deque(maxlen=100)

    for episode in range(1, n_episodes+1):
        current_state = env.reset()

        log_prob_list, reward_list = [], []
        for t in range(max_t):
            action, log_prob = agent.step(current_state)
            next_state, reward, done, _ = env.step(action)

            log_prob_list.append(log_prob)
            reward_list.append(reward)

            if episode % SHOW_EVERY == 0:
                env.render()
            if done:
                break
            else:
                current_state = next_state

        scores_deque.append(sum(reward_list))
        agent.learn(log_prob_list, reward_list, gamma)
        
        if episode % PRINT_EVERY == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_deque)))
        if np.mean(scores_deque)>=195.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode-100, np.mean(scores_deque)))
            torch.save(agent.alg.model.state_dict(), 'checkpoint2.pth')
            break

    env.close()

def test():
    env = gym.make('CartPole-v1')
    env.seed(0)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n 

    model = Model(state_dim, action_dim, seed=1)
    alg = PolicyGradient(model, device=device, lr=LR)
    agent = Agent(alg, state_dim, action_dim, device)
    agent.alg.model.load_state_dict(torch.load('checkpoint2.pth'))

    for i in range(10):
        state = env.reset()
        for j in range(150):
            action = agent.predict(state)
            state, reward, done, _ = env.step(action)
            env.render()
            if done:
                break   

    env.close()      


#train()
test()
