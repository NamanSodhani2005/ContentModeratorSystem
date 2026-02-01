"""
Target span toxicity model.
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel


class TargetSpanToxicityModel(nn.Module):
    """Token and toxicity heads."""

    # Initialize encoder and heads for token targets and toxicity classification.
    def __init__(self, model_name="distilbert-base-uncased", hidden_size=768, token_labels=2, num_tox_classes=2):
        super().__init__()
        self.num_tox_classes = num_tox_classes
        self.encoder = DistilBertModel.from_pretrained(model_name)  # shared encoder
        self.token_classifier = nn.Linear(hidden_size, token_labels)  # token head
        self.toxicity_classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_tox_classes)
        )  # toxicity head

        self.token_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)  # token loss
        self.tox_loss_fn = nn.CrossEntropyLoss()  # tox loss

    # Encode inputs, pool target representation, and compute toxicity logits and loss.
    def forward(self, input_ids, attention_mask, token_labels=None, toxicity_labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state

        token_logits = self.token_classifier(hidden)
        token_probs = torch.softmax(token_logits, dim=-1)[..., 1]  # target prob
        token_probs = token_probs * attention_mask  # mask padding

        denom = token_probs.sum(dim=1, keepdim=True).clamp(min=1e-6)  # stable sum
        weights = token_probs / denom  # normalize weights
        target_repr = torch.einsum("blh,bl->bh", hidden, weights)  # weighted target

        cls_repr = hidden[:, 0, :]  # CLS token
        combined = torch.cat([cls_repr, target_repr], dim=-1)  # concat features
        toxicity_logits = self.toxicity_classifier(combined)  # tox logits

        loss = None
        if token_labels is not None and toxicity_labels is not None:
            token_loss = self.token_loss_fn(token_logits.view(-1, token_logits.size(-1)), token_labels.view(-1))  # token loss
            tox_loss = self.tox_loss_fn(toxicity_logits, toxicity_labels)  # tox loss
            loss = tox_loss + 0.5 * token_loss  # joint loss

        return {
            "loss": loss,
            "token_logits": token_logits,
            "toxicity_logits": toxicity_logits
        }
