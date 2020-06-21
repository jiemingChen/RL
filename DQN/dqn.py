import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 

import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


from collections import namedtuple, deque
import numpy as np
import gym
from pdb import set_trace
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2

from ownenv import Blob, BlobEnv
import gym

AGGREGATE_STATS_EVERY = 50  # episodes
BUFFER_SIZE = int(1e5)  # replay buffer size
SHOW_PREVIEW = False
BATCH_SIZE = 32       # minibatch size
LR = 5e-4              # learning rate 
UPDATE_EVERY = 4       # how often to update the network
GAMMA = 0.99           # discount factor
TAU = 1e-3
LOAD_MODEL = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        #self.seed = random.seed(seed)
        random.seed(seed)
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class QNetwork(nn.Module):
	def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
		super(QNetwork, self).__init__()
		self.seed = torch.manual_seed(seed)
		self.fc1 = nn.Linear(state_size, fc1_units)
		self.fc2 = nn.Linear(fc1_units, fc2_units)
		self.fc3 = nn.Linear(fc2_units, action_size)

	def forward(self, state):
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		return self.fc3(x)

class DQNAgent:
	def __init__(self, state_size, action_size, seed=0):
		self.state_size = state_size
		self.action_size = action_size		
		#self.seed = random.seed(seed)

		#Q-Network
		self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
		self.qnetwork_target =  QNetwork(state_size, action_size, seed).to(device)
		if LOAD_MODEL is not False:
			self.qnetwork_local.load_state_dict(torch.load('checkpoint2.pth'))
			self.qnetwork_target.load_state_dict(torch.load('checkpoint2.pth'))

		self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

		# Replay memory
		self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

		#tensorboard
		self.tb = SummaryWriter()

		# Initialize time step (for updating every UPDATE_EVERY steps)
		self.t_step = 0  

	def act(self, state, eps=0.):
		state = torch.from_numpy(state).float().unsqueeze(0).to(device)
		self.qnetwork_local.eval()
		with torch.no_grad():
			action_values = self.qnetwork_local(state)
		self.qnetwork_local.train()

		if random.random() > eps:
			return np.argmax(action_values.cpu().data.numpy())
		else:
			return random.choice(np.arange(self.action_size))

	def step(self, state, action, reward, next_state, done):
		# Save experience in replay memory
		self.memory.add(state, action, reward, next_state, done)
	
		# Learn every UPDATE_EVERY time steps.
		self.t_step = (self.t_step+1) % UPDATE_EVERY
		if len(self.memory)>=BATCH_SIZE:
			experiences = self.memory.sample()
			self.learn(experiences, GAMMA)

	def learn(self, experiences, gamma):
		states, actions, rewards, next_states, dones = experiences		
		
		# Get max predicted Q values (for next states) from target model
		Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
		# Compute Q targets for current states 
		Q_targets = rewards + (gamma * Q_targets_next * (1-dones))
		# Get expected Q values from local model
		Q_expected = self.qnetwork_local(states).gather(1, actions)

		#Compute loss
		loss = F.mse_loss(Q_expected, Q_targets)
		#Compute grad and bp
		self.optimizer.zero_grad()
		loss.backward()
		#update target network
		self.optimizer.step()
		if self.t_step==0:
			self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

	def soft_update(self, local_model, target_model, tau):
		"""
			Soft update model parameters.
			θ_target = τ*θ_local + (1 - τ)*θ_target
			Params
			======
			local_model (PyTorch model): weights will be copied from
			target_model (PyTorch model): weights will be copied to
			tau (float): interpolation parameter 
		"""
		for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
			target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


def train(n_episodes=5000, max_t=200, eps_start=1.0, eps_end=0.01, eps_decay=0.996):
	scores = []
	scores_window = deque(maxlen=100)
	eps = eps_start

	for episode in range(1, n_episodes+1):
		current_state = env.reset() 
		episode_score = 0

		for t in range(max_t):
			action = agent.act(current_state, eps)
			new_state, reward, done, _  = env.step(action)

			agent.step(current_state, action, reward, new_state, done)

			if SHOW_PREVIEW and episode % AGGREGATE_STATS_EVERY==0:
				env.render()
			episode_score  += reward

			if done:
				break
			else:
				current_state =  new_state

		scores_window.append(episode_score)  # save most recent score
		scores.append(episode_score)  # save most recent score
		eps = max(eps_end, eps_decay * eps)  # decrease epsilon
		print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)), end="")
		if episode % 100 == 0:
			print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)))
		if np.min(scores_window)>=270.0 and len(scores_window)>=100:
			print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode - 100,
																						np.mean(scores_window)))
			torch.save(agent.qnetwork_local.state_dict(), 'checkpoint2.pth')
			break
		#print("!!!!!")
	torch.save(agent.qnetwork_local.state_dict(), 'checkpoint2.pth')


random.seed(1)
np.random.seed(1)
if not os.path.isdir("models"):
	os.makedirs("models")

env = BlobEnv()
agent = DQNAgent(state_size=env.OBSERVATION_SPACE_VALUES[0], action_size=env.ACTION_SPACE_SIZE)
#train()

agent.qnetwork_local.load_state_dict(torch.load('checkpoint2.pth'))
for i in range(100):
    state = env.reset()

    for j in range(25):
        action = agent.act(state)
        state, reward, done, _ = env.step(action)
        env.render()
        cv2.waitKey(50)
        if done:
            break 
