"""
DQN policy network.
"""

import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    """Q-network for moderation actions."""


    def __init__(self, comment_dim=768, context_dim=4, hidden_dim=256, num_actions=5):
        super().__init__()

        self.comment_dim = comment_dim
        self.context_dim = context_dim

        self.comment_processor = nn.Sequential(
            nn.Linear(comment_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.context_processor = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.q_network = nn.Sequential(
            nn.Linear(hidden_dim + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_actions)
        )


    def forward(self, state):
        comment_embedding = state[:, :self.comment_dim]
        context = state[:, self.comment_dim:]

        comment_features = self.comment_processor(comment_embedding)
        context_features = self.context_processor(context)

        combined = torch.cat([comment_features, context_features], dim=-1)
        q_values = self.q_network(combined)

        return q_values


    def get_action(self, state):
        with torch.no_grad():
            q_values = self.forward(state)
            action = torch.argmax(q_values, dim=-1)
        return action, q_values
