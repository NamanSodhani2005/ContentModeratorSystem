"""
DQN policy network with dual-stream architecture.

Two separate sub-networks process the comment embedding (768-dim) and
target/context features (4-dim) independently before merging. This lets
the network learn different representations for semantic content vs.
structured target signals, then combine them for the final Q-value estimate.
"""

import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    """Q-network for moderation actions.

    Architecture:
      comment_processor: 768 -> 256 -> 256  (semantic understanding)
      context_processor: 4 -> 64 -> 64      (target features: presence, hate/off/normal probs)
      q_network:         320 -> 128 -> 5    (merge streams -> Q-values for 5 actions)
    """

    def __init__(self, comment_dim=768, context_dim=4, hidden_dim=256, num_actions=5):
        super().__init__()

        self.comment_dim = comment_dim
        self.context_dim = context_dim

        # Stream 1: Process DistilBERT CLS embedding
        self.comment_processor = nn.Sequential(
            nn.Linear(comment_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        # Stream 2: Process target features (target_presence, hate/offensive/normal probs)
        self.context_processor = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True)
        )

        # Merge both streams and predict Q-values for each moderation action
        self.q_network = nn.Sequential(
            nn.Linear(hidden_dim + 64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_actions)
        )

    def forward(self, state):
        # Split state into comment embedding and context features
        comment_embedding = state[:, :self.comment_dim]
        context = state[:, self.comment_dim:]

        comment_features = self.comment_processor(comment_embedding)
        context_features = self.context_processor(context)

        combined = torch.cat([comment_features, context_features], dim=-1)
        q_values = self.q_network(combined)

        return q_values
