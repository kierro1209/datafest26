"""Small MLP policy for flat doctor×slot actions with external legal mask."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SchedulePolicyMLP(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def action_distribution(
        self,
        obs: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.distributions.Categorical:
        logits = self.forward(obs)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), -1e9)
        return torch.distributions.Categorical(logits=logits)
