"""
Attention Q-network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """Feature attention layer."""

    # Build attention projections to score features for interpretability.
    def __init__(self, input_dim, attention_dim=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, input_dim)
        )

    # Compute attention weights and reweight features for downstream scoring.
    def forward(self, x):
        scores = self.attention(x)
        weights = torch.softmax(scores, dim=-1)  # normalize weights
        attended = x * weights  # apply weights
        return attended, weights


class PolicyNetwork(nn.Module):
    """Q-network with attention."""

    # Configure encoders, attention block, and Q head for action scoring.
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
        )  # comment encoder

        self.context_processor = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )  # context encoder

        self.attention = AttentionLayer(hidden_dim + 64, attention_dim=128)  # attention block

        self.q_network = nn.Sequential(
            nn.Linear(hidden_dim + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_actions)
        )  # Q head

    # Encode inputs, apply attention, and return Q values with weights.
    def forward(self, state):
        comment_embedding = state[:, :self.comment_dim]  # comment slice
        context = state[:, self.comment_dim:]  # context slice

        comment_features = self.comment_processor(comment_embedding)
        context_features = self.context_processor(context)

        combined = torch.cat([comment_features, context_features], dim=-1)  # merge features

        attended, attention_weights = self.attention(combined)  # apply attention

        q_values = self.q_network(attended)  # Q values

        return q_values, attention_weights

    # Select greedy action and return Q values with attention weights.
    def get_action(self, state):
        with torch.no_grad():
            q_values, attention_weights = self.forward(state)
            action = torch.argmax(q_values, dim=-1)  # greedy action
        return action, q_values, attention_weights
