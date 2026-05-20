from pathlib import Path

import numpy as np
import torch
import warp as wp

from gpu_poker import constants as c
from gpu_poker.kernels import actions, legal_actions, observations, state
from gpu_poker.struct_types import GameState


@wp.kernel
def _fill_int_kernel(arr: wp.array(dtype=wp.int32), value: wp.int32):
    i = wp.tid()
    arr[i] = value


@wp.kernel
def _fill_reset_cfg_kernel(
    starting_stack_cfg: wp.array(dtype=wp.int32),
    small_blind_cfg: wp.array(dtype=wp.int32),
    big_blind_cfg: wp.array(dtype=wp.int32),
    starting_stack: wp.int32,
    small_blind: wp.int32,
    big_blind: wp.int32,
):
    i = wp.tid()
    starting_stack_cfg[i] = starting_stack
    small_blind_cfg[i] = small_blind
    big_blind_cfg[i] = big_blind


@wp.kernel
def step_kernel(
    game_state: GameState,
    action_types: wp.array(dtype=wp.int32),
    amounts: wp.array(dtype=wp.int32),
    rewards: wp.array(dtype=wp.float32),
    invalid_action: wp.array(dtype=wp.bool),
    terminated: wp.array(dtype=wp.bool),
    primes: wp.array(dtype=wp.int32),
    unsuited_lut: wp.array(dtype=wp.int16),
    flush_lut: wp.array(dtype=wp.int16),
    starting_stack: wp.int32,
    small_blind: wp.int32,
    big_blind: wp.int32,
):
    """Step kernel - processes one action per environment."""
    env_idx = wp.tid()

    prev_episode = game_state.episode_id[env_idx]

    action_type = action_types[env_idx]
    action_amount = amounts[env_idx]

    player = game_state.active_player[env_idx]
    packed = actions.force_legal_action(game_state, env_idx, player, action_type, action_amount)
    invalid_action[env_idx] = actions.unpack_invalid(packed)
    action_type = actions.unpack_action_type(packed)
    action_amount = actions.unpack_amount(packed)

    rewards[env_idx] = state.step(
        game_state,
        env_idx,
        action_type,
        action_amount,
        primes,
        unsuited_lut,
        flush_lut,
        starting_stack,
        small_blind,
        big_blind,
    )

    terminated[env_idx] = game_state.episode_id[env_idx] != prev_episode


@wp.kernel
def step_kernel_per_env(
    game_state: GameState,
    action_types: wp.array(dtype=wp.int32),
    amounts: wp.array(dtype=wp.int32),
    rewards: wp.array(dtype=wp.float32),
    invalid_action: wp.array(dtype=wp.bool),
    terminated: wp.array(dtype=wp.bool),
    primes: wp.array(dtype=wp.int32),
    unsuited_lut: wp.array(dtype=wp.int16),
    flush_lut: wp.array(dtype=wp.int16),
    starting_stack: wp.array(dtype=wp.int32),
    small_blind: wp.array(dtype=wp.int32),
    big_blind: wp.array(dtype=wp.int32),
):
    """Step kernel variant reading per-env reset parameters."""
    env_idx = wp.tid()

    prev_episode = game_state.episode_id[env_idx]

    action_type = action_types[env_idx]
    action_amount = amounts[env_idx]

    player = game_state.active_player[env_idx]
    packed = actions.force_legal_action(game_state, env_idx, player, action_type, action_amount)
    invalid_action[env_idx] = actions.unpack_invalid(packed)
    action_type = actions.unpack_action_type(packed)
    action_amount = actions.unpack_amount(packed)

    rewards[env_idx] = state.step(
        game_state,
        env_idx,
        action_type,
        action_amount,
        primes,
        unsuited_lut,
        flush_lut,
        starting_stack[env_idx],
        small_blind[env_idx],
        big_blind[env_idx],
    )

    terminated[env_idx] = game_state.episode_id[env_idx] != prev_episode


@wp.kernel
def get_obs_kernel(
    game_state: GameState,
    obs_cards: wp.array(dtype=wp.int32, ndim=2),
    obs_scalars: wp.array(dtype=wp.float32, ndim=2),
    primes: wp.array(dtype=wp.int32),
    unsuited_lut: wp.array(dtype=wp.int16),
    flush_lut: wp.array(dtype=wp.int16),
):
    """Observation kernel - extracts observations for all environments."""
    env_idx = wp.tid()

    observations.write_observation(
        game_state,
        env_idx,
        obs_cards,
        obs_scalars,
        primes,
        unsuited_lut,
        flush_lut,
    )


@wp.kernel
def get_obs_legal_kernel(
    game_state: GameState,
    obs_cards: wp.array(dtype=wp.int32, ndim=2),
    obs_scalars: wp.array(dtype=wp.float32, ndim=2),
    primes: wp.array(dtype=wp.int32),
    unsuited_lut: wp.array(dtype=wp.int16),
    flush_lut: wp.array(dtype=wp.int16),
    action_mask: wp.array(dtype=wp.bool, ndim=2),
    min_raise: wp.array(dtype=wp.int32),
    max_raise: wp.array(dtype=wp.int32),
):
    env_idx = wp.tid()
    observations.write_observation(
        game_state,
        env_idx,
        obs_cards,
        obs_scalars,
        primes,
        unsuited_lut,
        flush_lut,
    )
    legal_actions.write_legal_actions(game_state, env_idx, action_mask, min_raise, max_raise)


@wp.kernel
def get_legal_kernel(
    game_state: GameState,
    action_mask: wp.array(dtype=wp.bool, ndim=2),
    min_raise: wp.array(dtype=wp.int32),
    max_raise: wp.array(dtype=wp.int32),
):
    env_idx = wp.tid()
    legal_actions.write_legal_actions(game_state, env_idx, action_mask, min_raise, max_raise)


@wp.kernel
def clear_step_flags_kernel(
    invalid_action: wp.array(dtype=wp.bool),
    terminated: wp.array(dtype=wp.bool),
):
    i = wp.tid()
    invalid_action[i] = False
    terminated[i] = False


@wp.kernel
def reset_kernel(
    game_state: GameState,
    starting_stack: wp.int32,
    small_blind: wp.int32,
    big_blind: wp.int32,
):
    """Reset kernel - initializes all environments."""
    env_idx = wp.tid()
    state.reset_env(game_state, env_idx, starting_stack, small_blind, big_blind)


@wp.kernel
def reset_kernel_per_env(
    game_state: GameState,
    starting_stack: wp.array(dtype=wp.int32),
    small_blind: wp.array(dtype=wp.int32),
    big_blind: wp.array(dtype=wp.int32),
):
    """Reset kernel variant reading per-env reset parameters."""
    env_idx = wp.tid()
    state.reset_env(
        game_state,
        env_idx,
        starting_stack[env_idx],
        small_blind[env_idx],
        big_blind[env_idx],
    )


class WarpPokerEnv:
    """
    GPU-accelerated Heads-Up No-Limit Hold'em environment.

    Supports >100k parallel environments with zero-copy PyTorch integration.
    """

    # Device-scoped cache for large lookup tables (hundreds of MB on CUDA).
    # Without this, creating additional env instances (e.g. during evaluation)
    # can easily OOM even on large GPUs because each env would upload its own copy.
    _LOOKUP_CACHE: dict[str, tuple[wp.array, wp.array, wp.array]] = {}

    def __init__(
        self,
        num_envs: int = c.DEFAULT_NUM_ENVS,
        starting_stack: int = c.STARTING_STACK,
        small_blind: int = c.SMALL_BLIND,
        big_blind: int = c.BIG_BLIND,
        device: str = "cuda:0",
    ):
        self.num_envs = num_envs
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.device = device

        # 1. Allocate Global Arrays
        self.state = self._allocate_state()

        # 2. Load Lookup Tables (Cache on device)
        self.primes, self.unsuited_lut, self.flush_lut = self._load_lookups()

        # 3. Create Zero-Copy Tensor Views
        # These are what we return to PyTorch
        self.obs_cards = wp.zeros((num_envs, 7), dtype=wp.int32, device=device)
        self.obs_scalars = wp.zeros((num_envs, 17), dtype=wp.float32, device=device)
        self.rewards = wp.zeros(num_envs, dtype=wp.float32, device=device)
        self.dones = wp.zeros(num_envs, dtype=wp.bool, device=device)
        self.invalid_action = wp.zeros(num_envs, dtype=wp.bool, device=device)
        self.terminated = wp.zeros(num_envs, dtype=wp.bool, device=device)

        # Legal action mask + raise bounds (raise delta in chips)
        self.action_mask = wp.zeros((num_envs, c.NUM_ACTIONS), dtype=wp.bool, device=device)
        self.min_raise = wp.zeros(num_envs, dtype=wp.int32, device=device)
        self.max_raise = wp.zeros(num_envs, dtype=wp.int32, device=device)

        # 4. Action buffers
        self.actions = wp.zeros(num_envs, dtype=wp.int32, device=device)
        self.amounts = wp.zeros(num_envs, dtype=wp.int32, device=device)

        # 4b. Per-env reset parameters (used for stack/blind randomization).
        self.starting_stack_cfg = wp.zeros(num_envs, dtype=wp.int32, device=device)
        self.small_blind_cfg = wp.zeros(num_envs, dtype=wp.int32, device=device)
        self.big_blind_cfg = wp.zeros(num_envs, dtype=wp.int32, device=device)

        wp.launch(
            kernel=_fill_reset_cfg_kernel,
            dim=self.num_envs,
            inputs=[
                self.starting_stack_cfg,
                self.small_blind_cfg,
                self.big_blind_cfg,
                self.starting_stack,
                self.small_blind,
                self.big_blind,
            ],
            device=self.device,
        )

        # 5. Initialize all environments
        self.reset()

    def _allocate_state(self) -> GameState:
        """Allocates all backing arrays for GameState struct."""
        n = self.num_envs
        d = self.device

        # Game flow [N_ENVS]
        stage = wp.zeros(n, dtype=wp.int32, device=d)
        button = wp.zeros(n, dtype=wp.int32, device=d)
        active_player = wp.zeros(n, dtype=wp.int32, device=d)
        done = wp.zeros(n, dtype=wp.bool, device=d)

        # Deck
        deck = wp.zeros((n, c.NUM_CARDS), dtype=wp.int32, device=d)
        deck_top = wp.zeros(n, dtype=wp.int32, device=d)

        # Cards
        hole_cards = wp.zeros((n, c.NUM_PLAYERS, c.HOLE_CARDS), dtype=wp.int32, device=d)
        community_cards = wp.zeros((n, c.COMMUNITY_CARDS), dtype=wp.int32, device=d)
        num_community = wp.zeros(n, dtype=wp.int32, device=d)

        # Chips
        stacks = wp.zeros((n, c.NUM_PLAYERS), dtype=wp.int32, device=d)
        bets = wp.zeros((n, c.NUM_PLAYERS), dtype=wp.int32, device=d)
        initial_stacks = wp.zeros((n, c.NUM_PLAYERS), dtype=wp.int32, device=d)
        pot = wp.zeros(n, dtype=wp.int32, device=d)

        # Betting [N_ENVS]
        last_raise = wp.zeros(n, dtype=wp.int32, device=d)
        num_raises = wp.zeros(n, dtype=wp.int32, device=d)
        last_aggressor = wp.zeros(n, dtype=wp.int32, device=d)

        # Tracking [N_ENVS]
        num_actions = wp.zeros(n, dtype=wp.int32, device=d)
        actions_this_street = wp.zeros(n, dtype=wp.int32, device=d)
        last_action_type = wp.zeros(n, dtype=wp.int32, device=d)
        last_action_amount = wp.zeros(n, dtype=wp.int32, device=d)
        last_action_was_raise = wp.zeros(n, dtype=wp.int32, device=d)
        episode_id = wp.zeros(n, dtype=wp.int32, device=d)

        # Per-env configuration [N_ENVS]
        cfg_starting_stack = wp.zeros(n, dtype=wp.int32, device=d)
        cfg_small_blind = wp.zeros(n, dtype=wp.int32, device=d)
        cfg_big_blind = wp.zeros(n, dtype=wp.int32, device=d)

        # Initialize RNG seeds (different for each environment)
        # Use numpy to generate seeds, then create warp array
        seeds = np.random.randint(0, 2**32, size=n, dtype=np.uint32)
        rng_state = wp.array(seeds, dtype=wp.uint32, device=d)

        # Create struct and assign arrays
        game_state = GameState()
        game_state.stage = stage
        game_state.button = button
        game_state.active_player = active_player
        game_state.done = done
        game_state.deck = deck
        game_state.deck_top = deck_top
        game_state.hole_cards = hole_cards
        game_state.community_cards = community_cards
        game_state.num_community = num_community
        game_state.stacks = stacks
        game_state.bets = bets
        game_state.initial_stacks = initial_stacks
        game_state.pot = pot
        game_state.last_raise = last_raise
        game_state.num_raises = num_raises
        game_state.last_aggressor = last_aggressor
        game_state.num_actions = num_actions
        game_state.actions_this_street = actions_this_street
        game_state.last_action_type = last_action_type
        game_state.last_action_amount = last_action_amount
        game_state.last_action_was_raise = last_action_was_raise
        game_state.rng_state = rng_state
        game_state.episode_id = episode_id
        game_state.cfg_starting_stack = cfg_starting_stack
        game_state.cfg_small_blind = cfg_small_blind
        game_state.cfg_big_blind = cfg_big_blind

        return game_state

    def _load_lookups(self):
        """Loads lookup tables from disk and uploads to device."""
        cached = self._LOOKUP_CACHE.get(self.device)
        if cached is not None:
            return cached

        # Find lookup_data directory
        lookup_dir = Path(__file__).parent / "lookup_data"

        # Load hand ranks from single npz file
        hand_ranks = np.load(lookup_dir / "hand_ranks.npz")
        unsuited_np = hand_ranks["unsuited_table"].astype(np.int16)
        flush_np = hand_ranks["flush_table"].astype(np.int16)

        # Get primes from constants (already defined in constants.py)
        primes_np = np.array(c.RANK_PRIMES, dtype=np.int32)

        # Convert to Warp arrays (automatically uploads to device)
        primes = wp.array(primes_np, dtype=wp.int32, device=self.device)
        unsuited_lut = wp.array(unsuited_np, dtype=wp.int16, device=self.device)
        flush_lut = wp.array(flush_np, dtype=wp.int16, device=self.device)

        out = (primes, unsuited_lut, flush_lut)
        self._LOOKUP_CACHE[self.device] = out
        return out

    def reset(self):
        """Resets all environments to initial state."""
        wp.launch(
            kernel=clear_step_flags_kernel,
            dim=self.num_envs,
            inputs=[self.invalid_action, self.terminated],
            device=self.device,
        )

        wp.launch(
            kernel=reset_kernel_per_env,
            dim=self.num_envs,
            inputs=[
                self.state,
                self.starting_stack_cfg,
                self.small_blind_cfg,
                self.big_blind_cfg,
            ],
            device=self.device,
        )

        # Get initial observations
        wp.launch(
            kernel=get_obs_legal_kernel,
            dim=self.num_envs,
            inputs=[
                self.state,
                self.obs_cards,
                self.obs_scalars,
                self.primes,
                self.unsuited_lut,
                self.flush_lut,
                self.action_mask,
                self.min_raise,
                self.max_raise,
            ],
            device=self.device,
        )

        # Return initial observations
        return {
            "cards": wp.to_torch(self.obs_cards),
            "scalars": wp.to_torch(self.obs_scalars),
            "action_mask": wp.to_torch(self.action_mask),
            "min_raise": wp.to_torch(self.min_raise),
            "max_raise": wp.to_torch(self.max_raise),
            "terminated": wp.to_torch(self.terminated),
            "episode_id": wp.to_torch(self.state.episode_id),
            "player_id": wp.to_torch(self.state.active_player),
            "big_blind": wp.to_torch(self.big_blind_cfg),
            # Total chips committed this hand (pot + both current-street bets).
            # Used for discrete-bucket raise sizing (pot-fraction multipliers).
            "pot_total": (
                wp.to_torch(self.state.pot).to(dtype=torch.int32)
                + wp.to_torch(self.state.bets).sum(dim=-1).to(dtype=torch.int32)
            ),
        }

    def reset_with_config(
        self,
        *,
        starting_stack: int | torch.Tensor | None = None,
        small_blind: int | torch.Tensor | None = None,
        big_blind: int | torch.Tensor | None = None,
    ):
        """
        Reset with per-env parameters.

        Each parameter can be:
        - None (keep current config),
        - an int (broadcast to all envs),
        - or a torch int tensor of shape [num_envs].
        """
        if starting_stack is not None:
            self._set_cfg(self.starting_stack_cfg, starting_stack, dtype=torch.int32)
        if small_blind is not None:
            self._set_cfg(self.small_blind_cfg, small_blind, dtype=torch.int32)
        if big_blind is not None:
            self._set_cfg(self.big_blind_cfg, big_blind, dtype=torch.int32)
        return self.reset()

    def _set_cfg(self, dst: wp.array, value: int | torch.Tensor, *, dtype: torch.dtype) -> None:
        if isinstance(value, int):
            wp.launch(
                kernel=_fill_int_kernel,
                dim=self.num_envs,
                inputs=[dst, value],
                device=self.device,
            )
            return

        t = value.to(dtype=dtype, device=self.device)
        if t.ndim != 1 or t.shape[0] != self.num_envs:
            raise ValueError(f"Expected shape ({self.num_envs},), got {tuple(t.shape)}")
        wp.copy(dst, wp.from_torch(t, dtype=wp.int32))

    def step(self, actions_torch: torch.Tensor, amounts_torch: torch.Tensor = None):
        """
        Step all environments forward by one action.

        Args:
            actions_torch: [num_envs] tensor of action types (FOLD/CHECK/CALL/RAISE)
            amounts_torch: [num_envs] tensor of raise amounts (0 for non-raise actions)

        Returns:
            dict with keys: "cards", "scalars", "rewards", "dones"
        """
        # Handle amounts (default to 0 for non-raise actions)
        if amounts_torch is None:
            amounts_torch = torch.zeros_like(actions_torch)

        # Zero-copy convert torch -> warp
        wp.copy(self.actions, wp.from_torch(actions_torch, dtype=wp.int32))
        wp.copy(self.amounts, wp.from_torch(amounts_torch, dtype=wp.int32))

        # Launch Step Kernel
        wp.launch(
            kernel=step_kernel_per_env,
            dim=self.num_envs,
            inputs=[
                self.state,
                self.actions,
                self.amounts,
                self.rewards,
                self.invalid_action,
                self.terminated,
                self.primes,
                self.unsuited_lut,
                self.flush_lut,
                self.starting_stack_cfg,
                self.small_blind_cfg,
                self.big_blind_cfg,
            ],
            device=self.device,
        )

        # Launch Observation Kernel
        wp.launch(
            kernel=get_obs_legal_kernel,
            dim=self.num_envs,
            inputs=[
                self.state,
                self.obs_cards,
                self.obs_scalars,
                self.primes,
                self.unsuited_lut,
                self.flush_lut,
                self.action_mask,
                self.min_raise,
                self.max_raise,
            ],
            device=self.device,
        )

        # Copy done flags from state
        wp.copy(self.dones, self.state.done)

        # Return Torch Views (zero-copy)
        return {
            "cards": wp.to_torch(self.obs_cards),
            "scalars": wp.to_torch(self.obs_scalars),
            "rewards": wp.to_torch(self.rewards),
            "dones": wp.to_torch(self.dones),
            "invalid_action": wp.to_torch(self.invalid_action),
            "terminated": wp.to_torch(self.terminated),
            "episode_id": wp.to_torch(self.state.episode_id),
            "player_id": wp.to_torch(self.state.active_player),
            "action_mask": wp.to_torch(self.action_mask),
            "min_raise": wp.to_torch(self.min_raise),
            "max_raise": wp.to_torch(self.max_raise),
            "big_blind": wp.to_torch(self.big_blind_cfg),
            # Total chips committed this hand (pot + both current-street bets).
            # Used for discrete-bucket raise sizing (pot-fraction multipliers).
            "pot_total": (
                wp.to_torch(self.state.pot).to(dtype=torch.int32)
                + wp.to_torch(self.state.bets).sum(dim=-1).to(dtype=torch.int32)
            ),
        }

    def step_from_buffers(self):
        """
        Fast-path step that assumes `self.actions` and `self.amounts` are already populated
        on-device.

        This avoids torch->warp conversion and copies, and is intended for on-GPU action sampling
        and other advanced usage. If the buffers are stale/uninitialized, behavior is undefined.
        """
        wp.launch(
            kernel=step_kernel_per_env,
            dim=self.num_envs,
            inputs=[
                self.state,
                self.actions,
                self.amounts,
                self.rewards,
                self.invalid_action,
                self.terminated,
                self.primes,
                self.unsuited_lut,
                self.flush_lut,
                self.starting_stack_cfg,
                self.small_blind_cfg,
                self.big_blind_cfg,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=get_obs_legal_kernel,
            dim=self.num_envs,
            inputs=[
                self.state,
                self.obs_cards,
                self.obs_scalars,
                self.primes,
                self.unsuited_lut,
                self.flush_lut,
                self.action_mask,
                self.min_raise,
                self.max_raise,
            ],
            device=self.device,
        )

        wp.copy(self.dones, self.state.done)

        return {
            "cards": wp.to_torch(self.obs_cards),
            "scalars": wp.to_torch(self.obs_scalars),
            "rewards": wp.to_torch(self.rewards),
            "dones": wp.to_torch(self.dones),
            "invalid_action": wp.to_torch(self.invalid_action),
            "terminated": wp.to_torch(self.terminated),
            "episode_id": wp.to_torch(self.state.episode_id),
            "player_id": wp.to_torch(self.state.active_player),
            "action_mask": wp.to_torch(self.action_mask),
            "min_raise": wp.to_torch(self.min_raise),
            "max_raise": wp.to_torch(self.max_raise),
            "big_blind": wp.to_torch(self.big_blind_cfg),
            # Total chips committed this hand (pot + both current-street bets).
            # Used for discrete-bucket raise sizing (pot-fraction multipliers).
            "pot_total": (
                wp.to_torch(self.state.pot).to(dtype=torch.int32)
                + wp.to_torch(self.state.bets).sum(dim=-1).to(dtype=torch.int32)
            ),
        }
