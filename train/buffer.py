"""On-device rollout buffer for PPO training."""

from __future__ import annotations

import torch

from gpu_poker import constants as c


class RolloutBuffer:
    def __init__(
        self,
        *,
        t: int,
        n: int,
        scalar_dim: int,
        hidden: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.t = int(t)
        self.n = int(n)
        self.scalar_dim = int(scalar_dim)
        self.hidden = int(hidden)
        self.device = device
        self.dtype = dtype

        self.cards = torch.empty((t, n, 7), device=device, dtype=torch.int32)
        self.opp_cards = torch.empty((t, n, 2), device=device, dtype=torch.int32)
        self.scalars = torch.empty((t, n, scalar_dim), device=device, dtype=dtype)
        self.action_mask = torch.empty((t, n, c.NUM_ACTIONS), device=device, dtype=torch.bool)
        self.player_id = torch.empty((t, n), device=device, dtype=torch.int64)
        self.episode_id = torch.empty((t, n), device=device, dtype=torch.int32)
        self.reset_mask = torch.empty((t, n), device=device, dtype=torch.bool)
        self.learner_mask = torch.empty((t, n), device=device, dtype=torch.bool)

        self.action_type = torch.empty((t, n), device=device, dtype=torch.int64)
        self.raise_bucket = torch.empty((t, n), device=device, dtype=torch.int64)
        # Ratio-sensitive fields stay fp32 regardless of AMP dtype to avoid
        # quantizing log_pi_old away from replay-time log_pi_new.
        self.logprob = torch.empty((t, n), device=device, dtype=torch.float32)
        self.value = torch.empty((t, n), device=device, dtype=torch.float32)

        self.reward = torch.empty((t, n), device=device, dtype=torch.float32)
        self.reward_p0 = torch.empty((t, n), device=device, dtype=torch.float32)
        self.done = torch.empty((t, n), device=device, dtype=torch.bool)

        # Acting-player recurrent or packed-transformer state at action time.
        self.h = torch.empty((t, n, hidden), device=device, dtype=torch.float32)
        self.c = torch.empty((t, n, hidden), device=device, dtype=torch.float32)
