import torch
import numpy as np 
from collections import deque, namedtuple
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class Agent():
    def __init__(self, alg, buffer_size, batch_size, seed):
        self.alg = alg
        self.device = device
        self.memory = ReplayBuffer(buffer_size, batch_size, seed)
        self.batch_size = batch_size

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        self.alg.model.actor_model.eval()
        with torch.no_grad():
            action = self.alg.predict(state).cpu().detach().numpy()
        self.alg.model.actor_model.train()

        action = action.squeeze()
        action = np.clip(np.random.normal(action, 0.05), -1.0, 1.0)
        return action

    def step(self,  state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            states, actions, rewards, next_states, dones = experiences
            self.alg.learn(states, actions, rewards, next_states, dones)



