"""
Attention Q-network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """Feature attention layer."""


    def __init__(self, input_dim, attention_dim=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, input_dim)
        )


    def forward(self, x):
        scores = self.attention(x)
        weights = torch.softmax(scores, dim=-1)
        attended = x * weights
        return attended, weights


class PolicyNetwork(nn.Module):
    """Q-network with attention."""


    def __init__(self, comment_dim=768, context_dim=22, hidden_dim=256, num_actions=5):
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

        self.attention = AttentionLayer(hidden_dim + 64, attention_dim=128)

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

        attended, attention_weights = self.attention(combined)

        q_values = self.q_network(attended)

        return q_values, attention_weights


    def get_action(self, state):
        with torch.no_grad():
            q_values, attention_weights = self.forward(state)
            action = torch.argmax(q_values, dim=-1)
        return action, q_values, attention_weights
