"""
Hate speech head.
"""

import torch.nn as nn


class HateSpeechHead(nn.Module):
    """Hate/offensive classifier."""


    def __init__(self, input_dim=768, hidden_dim=128, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )


    def forward(self, embeddings):
        return self.net(embeddings)
