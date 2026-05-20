from __future__ import annotations

from dataclasses import dataclass

import torch
import warp as wp

from gpu_poker.env import WarpPokerEnv

try:
    from tensordict import TensorDict
    from torchrl.data import (
        Binary,
        Bounded,
        Categorical,
        Composite,
    )
    from torchrl.envs import EnvBase

    _HAS_TORCHRL = True
except ModuleNotFoundError:  # pragma: no cover
    EnvBase = object  # type: ignore[misc,assignment]
    _HAS_TORCHRL = False


@dataclass(frozen=True)
class TorchRLEnvConfig:
    num_envs: int
    starting_stack: int
    small_blind: int
    big_blind: int
    device: str


class WarpPokerTorchRLEnv(EnvBase):
    """
    TorchRL wrapper around `WarpPokerEnv`.

    Keys:
    - observations: `cards`, `scalars`, `action_mask`, `min_raise`, `max_raise`,
      `episode_id`, `terminated`
    - actions: `action_type` (0..3), `raise_frac` (float in [0, 1])
    - outputs: `reward`, `done` (mapped from `terminated`)
    """

    def __init__(
        self,
        *,
        num_envs: int,
        device: str = "cuda:0",
        starting_stack: int = 1000,
        small_blind: int = 1,
        big_blind: int = 2,
    ):
        if not _HAS_TORCHRL:  # pragma: no cover
            raise ModuleNotFoundError(
                "torchrl is not installed. Install torchrl+tensordict to use WarpPokerTorchRLEnv."
            )

        self.cfg = TorchRLEnvConfig(
            num_envs=num_envs,
            starting_stack=starting_stack,
            small_blind=small_blind,
            big_blind=big_blind,
            device=device,
        )
        self._torch_device = torch.device(device if device != "cuda" else "cuda:0")
        self._env = WarpPokerEnv(
            num_envs=num_envs,
            device=device,
            starting_stack=starting_stack,
            small_blind=small_blind,
            big_blind=big_blind,
        )
        self._last_action_mask = None
        self._last_min_raise = None
        self._last_max_raise = None

        super().__init__(device=self._torch_device, batch_size=torch.Size([num_envs]))
        self._make_specs()

    def _make_specs(self) -> None:
        n = self.cfg.num_envs
        d = self._torch_device

        self.observation_spec = Composite(
            cards=Bounded(
                low=-1,
                high=51,
                shape=(n, 7),
                dtype=torch.int32,
                device=d,
            ),
            scalars=Bounded(
                low=0.0,
                high=1.0,
                shape=(n, 17),
                dtype=torch.float32,
                device=d,
            ),
            player_id=Categorical(
                n=2,
                shape=(n,),
                dtype=torch.int32,
                device=d,
            ),
            action_mask=Binary(
                shape=(n, 4),
                dtype=torch.bool,
                device=d,
            ),
            min_raise=Bounded(
                low=0,
                high=self.cfg.starting_stack,
                shape=(n,),
                dtype=torch.int32,
                device=d,
            ),
            max_raise=Bounded(
                low=0,
                high=self.cfg.starting_stack,
                shape=(n,),
                dtype=torch.int32,
                device=d,
            ),
            episode_id=Bounded(
                low=0,
                high=2**31 - 1,
                shape=(n,),
                dtype=torch.int32,
                device=d,
            ),
            shape=(n,),
        )

        self.action_spec = Composite(
            action_type=Categorical(
                n=4,
                shape=(n,),
                dtype=torch.int32,
                device=d,
            ),
            amount=Bounded(
                low=0,
                high=self.cfg.starting_stack,
                shape=(n,),
                dtype=torch.int32,
                device=d,
            ),
            raise_frac=Bounded(low=0.0, high=1.0, shape=(n, 1), dtype=torch.float32, device=d),
            shape=(n,),
        )

        self.reward_spec = Bounded(
            low=-float(self.cfg.starting_stack),
            high=float(self.cfg.starting_stack),
            shape=(n, 1),
            dtype=torch.float32,
            device=d,
        )
        self.done_spec = Binary(shape=(n, 1), dtype=torch.bool, device=d)

    def _set_seed(self, seed: int | None) -> None:
        if seed is None:
            return

        @wp.kernel
        def seed_kernel(rng_state: wp.array(dtype=wp.uint32), base: wp.uint32):
            i = wp.tid()
            # Simple per-env hash (Knuth multiplicative).
            rng_state[i] = base ^ (wp.uint32(i) * wp.uint32(2654435761))

        wp.launch(
            seed_kernel,
            dim=self.cfg.num_envs,
            inputs=[self._env.state.rng_state, wp.uint32(seed)],
            device=self.cfg.device,
        )

    def _reset(self, tensordict: TensorDict | None = None, **kwargs) -> TensorDict:
        out = self._env.reset()
        self._last_action_mask = out["action_mask"]
        self._last_min_raise = out["min_raise"]
        self._last_max_raise = out["max_raise"]
        return TensorDict(
            {
                "cards": out["cards"],
                "scalars": out["scalars"],
                "player_id": out["player_id"],
                "action_mask": out["action_mask"],
                "min_raise": out["min_raise"],
                "max_raise": out["max_raise"],
                "episode_id": out["episode_id"],
            },
            batch_size=self.batch_size,
            device=self._torch_device,
        )

    def _step(self, tensordict: TensorDict) -> TensorDict:
        if "action" in tensordict:
            action_td = tensordict["action"]
            action_type = action_td["action_type"]
            amount = action_td.get("amount", None)
            raise_frac = action_td.get("raise_frac", None)
        else:
            action_type = tensordict["action_type"]
            amount = tensordict.get("amount", None)
            raise_frac = tensordict.get("raise_frac", None)

        action_type_i32 = action_type.to(dtype=torch.int32, device=self._torch_device)

        # Fast path: if a precomputed `amount` is provided, use it directly.
        if amount is not None:
            amount_i32 = amount.to(dtype=torch.int32, device=self._torch_device)
            out = self._env.step(action_type_i32, amount_i32)
            self._last_action_mask = out["action_mask"]
            self._last_min_raise = out["min_raise"]
            self._last_max_raise = out["max_raise"]

            terminated = out["terminated"].unsqueeze(-1)
            reward = out["rewards"].unsqueeze(-1)

            next_td = TensorDict(
                {
                    "cards": out["cards"],
                    "scalars": out["scalars"],
                    "player_id": out["player_id"],
                    "action_mask": out["action_mask"],
                    "min_raise": out["min_raise"],
                    "max_raise": out["max_raise"],
                    "terminated": terminated,
                    "episode_id": out["episode_id"],
                    "reward": reward,
                    "done": terminated,
                    "invalid_action": out["invalid_action"],
                },
                batch_size=self.batch_size,
                device=self._torch_device,
            )
            return next_td

        # Map raise_frac -> raise delta (chips) using cached bounds.
        # This keeps the action interface stable even if the caller only provides actions.
        if raise_frac is None:
            raise_frac = torch.zeros(
                (self.cfg.num_envs, 1), dtype=torch.float32, device=self._torch_device
            )
        else:
            raise_frac = raise_frac.to(dtype=torch.float32, device=self._torch_device)

        if self._last_min_raise is None or self._last_max_raise is None:
            raise ValueError("Env must be reset before stepping.")

        min_r = self._last_min_raise.to(dtype=torch.int32, device=self._torch_device)
        max_r = self._last_max_raise.to(dtype=torch.int32, device=self._torch_device)

        frac = torch.clamp(raise_frac.squeeze(-1), 0.0, 1.0)
        span = torch.clamp(max_r - min_r, min=0)
        amount = min_r + torch.floor(frac * span.to(torch.float32)).to(torch.int32)

        # Only apply amount on raises; otherwise zero.
        amount = torch.where(action_type_i32 == 3, amount, torch.zeros_like(amount))

        out = self._env.step(action_type_i32, amount)
        self._last_action_mask = out["action_mask"]
        self._last_min_raise = out["min_raise"]
        self._last_max_raise = out["max_raise"]

        terminated = out["terminated"].unsqueeze(-1)
        reward = out["rewards"].unsqueeze(-1)

        next_td = TensorDict(
            {
                "cards": out["cards"],
                "scalars": out["scalars"],
                "player_id": out["player_id"],
                "action_mask": out["action_mask"],
                "min_raise": out["min_raise"],
                "max_raise": out["max_raise"],
                "terminated": terminated,
                "episode_id": out["episode_id"],
                "reward": reward,
                "done": terminated,
                "invalid_action": out["invalid_action"],
            },
            batch_size=self.batch_size,
            device=self._torch_device,
        )
        return next_td

    def step_from_env_buffers(self) -> TensorDict:
        """
        Fast-path for benchmarks: assumes the underlying `WarpPokerEnv.actions/amounts`
        buffers are already populated on-device, and steps without torch->warp copies.
        """
        out = self._env.step_from_buffers()
        self._last_action_mask = out["action_mask"]
        self._last_min_raise = out["min_raise"]
        self._last_max_raise = out["max_raise"]

        terminated = out["terminated"].unsqueeze(-1)
        reward = out["rewards"].unsqueeze(-1)

        return TensorDict(
            {
                "cards": out["cards"],
                "scalars": out["scalars"],
                "player_id": out["player_id"],
                "action_mask": out["action_mask"],
                "min_raise": out["min_raise"],
                "max_raise": out["max_raise"],
                "terminated": terminated,
                "episode_id": out["episode_id"],
                "reward": reward,
                "done": terminated,
                "invalid_action": out["invalid_action"],
            },
            batch_size=self.batch_size,
            device=self._torch_device,
        )
