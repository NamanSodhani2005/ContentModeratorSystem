"""
Attention-based Q-Network for content moderation.
"""

import torch # PyTorch framework
import torch.nn as nn # Neural network modules
import torch.nn.functional as F # Functional operations

class AttentionLayer(nn.Module): # Attention layer class
    """Attention mechanism for interpretability."""

    def __init__(self, input_dim, attention_dim=128): # Initialize attention
        super().__init__() # Call parent constructor
        self.attention = nn.Sequential( # Attention network
            nn.Linear(input_dim, attention_dim), # First projection
            nn.Tanh(), # Nonlinearity
            nn.Linear(attention_dim, input_dim) # Per-feature scores
        )

    def forward(self, x): # Forward pass
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            attended: (batch_size, input_dim)
            weights: (batch_size, input_dim) - attention weights
        """
        scores = self.attention(x) # Compute attention scores
        weights = torch.softmax(scores, dim=-1) # Normalize across features
        attended = x * weights # Apply attention
        return attended, weights # Return results

class PolicyNetwork(nn.Module): # Q-network class
    """
    Q-Network with attention for content moderation.

    Architecture:
    - Comment processor: 768 → 256
    - Context processor: 18 → 64
    - Attention layer for interpretability
    - Q-value heads: 320 → 5 actions
    """

    def __init__(self, comment_dim=768, context_dim=18, hidden_dim=256, num_actions=5): # Initialize network
        super().__init__() # Call parent constructor

        self.comment_dim = comment_dim # Comment embedding size
        self.context_dim = context_dim # Context vector size

        # Comment embedding processor
        self.comment_processor = nn.Sequential( # Comment encoder
            nn.Linear(comment_dim, hidden_dim), # Project to hidden
            nn.ReLU(), # Activation
            nn.Dropout(0.3), # Regularization
            nn.Linear(hidden_dim, hidden_dim), # Second layer
            nn.ReLU() # Activation
        )

        # Context processor (user history + platform metrics)
        self.context_processor = nn.Sequential( # Context encoder
            nn.Linear(context_dim, 64), # Project to 64
            nn.ReLU(), # Activation
            nn.Linear(64, 64), # Second layer
            nn.ReLU() # Activation
        )

        # Attention layer
        self.attention = AttentionLayer(hidden_dim + 64, attention_dim=128) # Attention mechanism

        # Q-value heads
        self.q_network = nn.Sequential( # Q-value network
            nn.Linear(hidden_dim + 64, 128), # Project to 128
            nn.ReLU(), # Activation
            nn.Dropout(0.2), # Regularization
            nn.Linear(128, num_actions) # Output Q-values
        )

    def forward(self, state): # Forward pass
        """
        Args:
            state: (batch_size, 786) - [comment(768), context(18)]
        Returns:
            q_values: (batch_size, 5)
            attention_weights: (batch_size, 320) - per-feature weights
        """
        # Split state into components
        comment_embedding = state[:, :self.comment_dim] # Extract comment part
        context = state[:, self.comment_dim:] # Extract context part

        # Process components
        comment_features = self.comment_processor(comment_embedding) # Encode comment
        context_features = self.context_processor(context) # Encode context

        # Combine features
        combined = torch.cat([comment_features, context_features], dim=-1) # Concatenate features

        # Apply attention
        attended, attention_weights = self.attention(combined) # Apply attention mechanism

        # Compute Q-values
        q_values = self.q_network(attended) # Compute action values

        return q_values, attention_weights # Return outputs

    def get_action(self, state): # Select best action
        """Select action with highest Q-value."""
        with torch.no_grad(): # No gradient needed
            q_values, attention_weights = self.forward(state) # Forward pass
            action = torch.argmax(q_values, dim=-1) # Select max Q
        return action, q_values, attention_weights # Return action and values
