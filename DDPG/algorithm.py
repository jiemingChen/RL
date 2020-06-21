import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np 
from pdb import set_trace
import copy 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPG():
    def __init__(self, model, target_model, gamma=None, tau=None, actor_lr=None, critic_lr=None):
        """  DDPG algorithm

        Args:
            model (nn.Module):
            gamma (float): reward decaying rate
            tau (float): self.target_model soft update rate
            actor_lr (float): actor learning rate
            critic_lr (float): critic learning rate
            device : cpu or gpu
        """
        assert isinstance(gamma, float)
        assert isinstance(tau, float)
        assert isinstance(actor_lr, float)
        assert isinstance(critic_lr, float)

        self.device = device
        self.gamma = gamma
        self.tau = tau

        self.model = model.to(self.device)
        self.target_model = target_model.to(self.device)  # check separate model !!!

        self.actor_optimizer = optim.Adam(self.model.actor_model.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.model.critic_model.parameters(), lr=critic_lr)


    def predict(self, states):
        return self.model.policy(states)

    def learn(self, states, actions, rewards, next_states, dones):
        critic_loss = self._critic_learn(states, actions, rewards, next_states, dones)
        actor_loss = self._actor_learn(states)
        self._soft_update(self.model.actor_model, self.target_model.actor_model)
        self._soft_update(self.model.critic_model, self.target_model.critic_model)


    def _critic_learn(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_actions = self.target_model.policy(next_states)
            next_Q = self.target_model.value(next_states, next_actions)
        Q_target = rewards + self.gamma * next_Q * (1 - dones)

        Q_expected = self.model.value(states, actions)
        
        critic_loss = F.mse_loss(Q_expected, Q_target)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return critic_loss

    def _actor_learn(self, states):
        actions = self.model.policy(states)
        Q = self.model.value(states, actions)

        actor_loss = -Q.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()     

        return  actor_loss


    def _soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
