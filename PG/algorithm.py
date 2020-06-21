import torch
import torch.optim as optim
from pdb import set_trace

class PolicyGradient():
    def __init__(self, model, device, lr=None):
        """ Policy Gradient algorithm
        Args：
            model(nn.Moduel): policy (inference network)
            lr(float): learning rate
        """
        self.model = model.to(device)
        assert isinstance(lr, float)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.device = device
    
    def predict(self, state):
        """ inference one time to get action probs
        """
        act_probs = self.model(state)
        return act_probs.cpu()
    
    def inference(self, state):
        self.model.eval()
        with torch.no_grad():
            act_probs = self.model(state)
        self.model.train() 

        return act_probs.cpu()

    def learn(self, log_prob_list, reward_list, gamma):
        """ policy gradient to update policy model
        """
        #Compute loss
        gains_list = reward_list
        # G_t = r_t + γ·r_t+1 + ... = r_t + γ·G_t+1
        for i in range(len(reward_list)-1, -1, -1):
            gains_list[i-1] = reward_list[i-1] + gamma*gains_list[i]

        gains = torch.tensor(gains_list).to(self.device).detach()
        log_probs = torch.cat(log_prob_list).to(self.device)

        cost = -1.0 *  gains * log_probs
        cost = cost.mean()
        #Compute grad and bp
        self.optimizer.zero_grad()
        cost.backward()
        #update model
        self.optimizer.step()
    
        return cost

