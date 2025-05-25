import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim=18, action_dim=3, hidden_sizes=[64, 64],
                 activation=nn.ELU, initial_log_std=0.0):
        super().__init__()

        # Define MLP network
        self.net_container = nn.Sequential(
            nn.Linear(state_dim, hidden_sizes[0]),
            activation(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            activation(),
        )

        self.policy_layer = nn.Linear(hidden_sizes[1], action_dim)

        self.value_layer = nn.Linear(hidden_sizes[1], 1)

        # Learnable log standard deviation parameter
        self.log_std_parameter = nn.Parameter(torch.ones(action_dim) * initial_log_std)

    def forward(self, states):
        x = self.net_container(states)
        policy = self.policy_layer(x)
        value = self.value_layer(x)
        std = torch.exp(self.log_std_parameter)
        return policy, value, std