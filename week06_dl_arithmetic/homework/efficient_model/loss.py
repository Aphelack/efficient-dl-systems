"""
Cross Entropy Loss for Causal LM
"""

import torch
import torch.nn as nn
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

class CrossEntropyLoss(nn.Module):
    """Fused Linear Cross Entropy for causal LM."""
    # TODO: Replace with fused linear cross entropy (LigerFusedLinearCrossEntropyLoss)
    # The fused version takes hidden_states + lm_head.weight instead of logits

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, hidden_states: torch.Tensor, lm_head_weight: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_hidden_states = hidden_states[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten tokens
        shift_hidden_states = shift_hidden_states.view(-1, self.config.hidden_size)
        shift_labels = shift_labels.view(-1)

        lce = LigerFusedLinearCrossEntropyLoss(reduction="mean")
        loss = lce(lm_head_weight, shift_hidden_states, shift_labels)
        return loss
