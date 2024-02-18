import torch
import numpy as np
import torch.nn as nn
from torch.distributions import Categorical

class ACNET(nn.Module):
    
    def __init__(self, n_actions, feature_space):
        super(ACNET, self).__init__()
        self.n_actions = n_actions
        self.HIDDEN = 256
        self.layer1 = nn.Linear(in_features=feature_space, out_features= self.HIDDEN)
        self.layer2 = nn.Linear(in_features= self.HIDDEN, out_features= self.HIDDEN)
        self.ReLU = nn.ReLU()
        self.value = nn.Linear(in_features = self.HIDDEN, out_features=1)
        self.pi = nn.Linear(in_features= self.HIDDEN, out_features= n_actions)
        self.softmax = nn.Softmax(dim = -1)
        
    def forward(self, x):
        x =  self.layer2(self.ReLU(self.layer2(self.ReLU(self.layer1(x)))))
        value = self.value(x)
        pi = self.softmax(self.pi(x))
        
        return value, pi

class Agent:
    
    def __init__(self, n_actions, feature_space):
        self.discout_factor = 1.0
        self.lr = 5e-1
        self.n_actions = n_actions
        self.prev_action = None
        self.action_space = np.arange(self.n_actions)
        self.network = ACNET(n_actions=n_actions, feature_space=feature_space)
        self.optimizer = torch.optim.Adagrad(params=self.network.parameters(), lr=self.lr, lr_decay=1e-3)
        
    
    
    def action_selection(self, observation):
        observation = torch.tensor(observation)
        with torch.inference_mode():   
            value, probs = self.network(observation)
            action_probs = Categorical(probs=probs)
            sampled_action = action_probs.sample()
            self.prev_action = sampled_action
            return sampled_action.item()
    
    def learn(self, state, reward, new_state, done):
        state = torch.tensor(state)
        new_state = torch.tensor(new_state)
        reward = torch.tensor(reward)
        
        # FORWARD PASS
        self.network.train()
        state_val, probs = self.network(state)
        new_state_val, _ = self.network(new_state)
        
        # calculate the loss
        action_probs = Categorical(probs=probs)
        log_probs = action_probs.log_prob(self.prev_action)
        
        delta = reward + new_state_val*self.discount_factor*(1-int(done)) - state_val
        
        critic_loss = delta**2
        actor_loss = delta*log_probs
        
        # reset grad value to None
        self.optimizer.zero_grad()
        
        # bakprop the losses
        critic_loss.backward(retain_graph=True)
        actor_loss.backward(retain_graph=True)
        
        # update the weights
        self.optimizer.step()
                