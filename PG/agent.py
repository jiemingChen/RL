import numpy as np 
import torch
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from pdb import set_trace
class Agent():
    def __init__(self, alg, state_size, action_size, device):
        self.state_size = state_size
        self.action_size = action_size
        self.alg = alg

        self.device = device
        #self.tb = SummaryWriter()
        self.cnt =0

    def step(self, state):
        """ output random policy
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        #if self.cnt==0:
        #    self.tb.add_graph(self.alg.model, state)
        #    self.cnt = 1

        act_probs = self.alg.predict(state).squeeze()

        act =  np.random.choice(range(self.action_size), p=act_probs.detach().numpy())
        log_prob = torch.log(act_probs[act])

        return act, log_prob.unsqueeze(0)

    def learn(self, log_prob_list, reward_list, gamma=1.0):
        loss = self.alg.learn(log_prob_list, reward_list, gamma)


    def predict(self, state):
        """ output determined policy
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        act_probs = self.alg.inference(state).squeeze()
        act = np.argmax(act_probs)
        return act.cpu().numpy()



