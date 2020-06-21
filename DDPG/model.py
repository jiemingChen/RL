import torch
import torch.nn as nn
import torch.nn.functional as F 

class Model(nn.Module):
    def __init__(self, state_size, action_size, seed=0):
        super(Model, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.actor_model = ActorModel(state_size, action_size, seed)
        self.critic_model =  CriticModel(state_size, action_size, seed)

    def policy(self, states):
        """
        should be batch states torch
        """
        actions = self.actor_model(states)
        return actions

    def value(self, state, action):
        """
        state, action: torch
        """
        Q = self.critic_model(state, action)
        return Q
    """
    def get_actor_params(self):
        # return parameter names
        
        return self.actor_model.parameters() #!!!
    """

class CriticModel(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=256, fc3_units=128):
        """ estimate Q value
        """
        super(CriticModel, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)



    def forward(self, state, action):
        """
        Params:
        input: [state, action]
        """
        xs = F.leaky_relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return self.fc4(x)


class ActorModel(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=256):
        super(ActorModel, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units) 
        self.fc2 = nn.Linear(fc1_units, action_size)    

    def forward(self, state):
        x  = F.relu(self.fc1(state))
        return torch.tanh(self.fc2(x))        

   