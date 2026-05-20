from __future__ import annotations

import contextlib
import copy
import json
import os
import socket
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from tqdm import tqdm

# Allow running `uv run train/train.py` (script mode) while still importing `train.*`.
# When executed as a script, Python puts `.../repo/train` on `sys.path` first, which
# can cause `import train.*` to resolve to `train/train.py` (a module) instead of
# the `train/` package. Force the repo root to be first on `sys.path`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_str = str(_REPO_ROOT)
if not sys.path or sys.path[0] != repo_root_str:
    sys.path.insert(0, repo_root_str)

from gpu_poker import constants as c  # noqa: E402
from gpu_poker.env import WarpPokerEnv  # noqa: E402
from gpu_poker.policy import (  # noqa: E402
    NUM_RAISE_BUCKETS,
    PokerPolicyNet,
    raise_bucket_to_amount,
)
from gpu_poker.policy_transformer import (  # noqa: E402
    PokerTransformerPolicyNet,
    push_action_token_packed,
)

try:
    from train import bots as train_bots  # type: ignore[attr-defined]
    from train.eval import (  # type: ignore[attr-defined]
        eval_vs_bot,
        eval_vs_bot_seat_swap,
        eval_vs_policy,
        eval_vs_snapshot_mix_seat_swap,
    )
except ModuleNotFoundError:  # pragma: no cover
    # Fallback for environments where `train` resolves incorrectly.
    import bots as train_bots  # type: ignore[no-redef]
    from eval import (  # type: ignore[no-redef]
        eval_vs_bot,
        eval_vs_bot_seat_swap,
        eval_vs_policy,
        eval_vs_snapshot_mix_seat_swap,
    )

from train.checkpointing import CheckpointState, load_checkpoint, save_checkpoint  # noqa: E402
from train.league import (  # noqa: E402
    OpponentMeta,
    UnifiedOpponentPool,
    pfsp_weights_from_bb_per_hand,
)
from train.triad import (  # noqa: E402
    OPP_LIVE_MAIN,
    OPP_SELF,
    OPP_SNAPSHOT,
    ROLE_LE,
    ROLE_MAIN,
    ROLE_ME,
    ROLES,
    TriadController,
)

try:
    from torch.utils.tensorboard import SummaryWriter

    _HAS_TB = True
except ModuleNotFoundError:  # pragma: no cover
    SummaryWriter = None  # type: ignore[assignment]
    _HAS_TB = False

try:
    import pynvml

    _HAS_PYNVML = True
except ModuleNotFoundError:
    pynvml = None
    _HAS_PYNVML = False

try:
    import hydra
    from hydra.core.config_store import ConfigStore
    from omegaconf import OmegaConf

    _HAS_HYDRA = True
except ModuleNotFoundError:  # pragma: no cover
    hydra = None  # type: ignore[assignment]
    ConfigStore = None  # type: ignore[assignment]
    OmegaConf = None  # type: ignore[assignment]
    _HAS_HYDRA = False


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _debug_numeric_enabled() -> bool:
    return os.environ.get("POKERGPU_DEBUG_NUMERIC", "0") == "1"


def _assert_finite(name: str, x: torch.Tensor) -> None:
    if not _debug_numeric_enabled():
        return
    if torch.isfinite(x).all():
        return
    bad = (~torch.isfinite(x)).sum().item()
    raise RuntimeError(f"Non-finite tensor: {name} (bad={bad}, shape={tuple(x.shape)})")


# Either policy backbone. Both expose the same forward_step / evaluate_step /
# lstm_hidden contract, so the rest of the trainer treats them interchangeably.
PolicyNet = PokerPolicyNet | PokerTransformerPolicyNet


def _make_policy(cfg: TrainConfig, scalar_dim: int, device: torch.device) -> PolicyNet:
    """Construct the configured policy backbone. Both backbones expose the
    same forward_step / evaluate_step contract and a `.lstm_hidden` int that
    train.py uses as the trailing dim for state buffers (LSTM hidden size for
    "lstm"; packed STATE_WIDTH for "transformer").
    """
    backbone = (cfg.policy_backbone or "lstm").lower()
    if backbone == "lstm":
        return PokerPolicyNet(
            scalar_dim=scalar_dim,
            card_embed_dim=cfg.card_embed_dim,
            mlp_dim=cfg.mlp_dim,
            torso_layers=cfg.torso_layers,
            lstm_hidden=cfg.lstm_hidden,
            head_layers=cfg.head_layers,
            head_dim=cfg.head_dim,
        ).to(device=device)
    if backbone == "transformer":
        return PokerTransformerPolicyNet(
            scalar_dim=scalar_dim,
            card_embed_dim=cfg.card_embed_dim,
            d_model=cfg.transformer_d_model,
            n_heads=cfg.transformer_n_heads,
            n_layers=cfg.transformer_n_layers,
            ffn_dim=cfg.transformer_ffn_dim,
        ).to(device=device)
    raise ValueError(f"Unknown policy_backbone: {cfg.policy_backbone!r}")


def load_snapshot_model(
    path: str, device: torch.device, cfg: TrainConfig, scalar_dim: int
) -> PolicyNet:
    """Robustly load a snapshot model, handling potential torch.compile prefixes.

    Loads strictly: a shape or key mismatch raises immediately. Stale snapshots
    (e.g., from before a model-architecture change) will fail with a clear
    shape-mismatch error from PyTorch wrapped with a hint to clear the pool.
    """
    ckpt = torch.load(path, map_location=device)
    # Clean state dict if it's from a compiled model
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}

    net = _make_policy(cfg, scalar_dim, device)
    try:
        net.load_state_dict(state_dict)
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to load snapshot at {path}; the architecture likely "
            f"changed since this snapshot was saved. Clear stale snapshots "
            f"from the league pool and resume."
        ) from e
    net.eval()
    return net


@dataclass(frozen=True)
class TrainConfig:
    device: str
    num_envs: int
    rollout_steps: int
    # Termination: prefer `total_env_steps` (set >0). `num_updates` is kept for
    # backward-compatibility but is not recommended.
    total_env_steps: int
    num_updates: int
    lr: float
    gamma: float
    gae_lambda: float
    clip_eps: float
    value_coef: float
    entropy_coef: float
    max_grad_norm: float
    # Policy architecture
    card_embed_dim: int
    mlp_dim: int
    torso_layers: int
    lstm_hidden: int
    head_layers: int
    head_dim: int | None
    num_epochs: int
    minibatch_size: int
    seed: int
    # Logging / progress
    log_every_env_steps: int
    log_every_seconds: float
    # Checkpointing
    checkpoint_dir: str
    save_every_env_steps: int
    resume_from: str
    eval_every: int
    eval_hands: int
    eval_max_steps: int
    # League / opponent sampling (NFSP-lite). When enabled, training collects experience
    # from learner-controlled actions against a mixture of opponents (bots + snapshot pool).
    # It uses Monte-Carlo terminal returns (no GAE) to avoid leaking opponent hole cards
    # through the critic on opponent-to-act timesteps (env observations are active-player).
    league_enabled: bool
    # Mixture policy:
    # - with prob `league_eta_selfplay`, play vs the latest policy (self-play)
    # - otherwise play vs a PFSP-weighted bot/snapshot from the unified pool
    league_eta_selfplay: float
    # Initial bb/hand for bots (0 = assume 50% win rate)
    league_bot_initial_score: float
    # Whether to evaluate vs bots for PFSP
    league_eval_bots: bool
    # How often to re-evaluate bots (in updates)
    league_eval_bots_every: int
    # Disk-backed snapshot pool (stored under `checkpoint_dir/league_pool`).
    league_snapshot_pool: int
    league_snapshot_every_updates: int
    league_eval_every_updates: int
    league_eval_hands: int
    league_eval_subset: int
    league_eval_deterministic: bool
    league_eval_seat_swap: bool
    league_pfsp_temperature: float
    league_pfsp_epsilon: float
    # Rollout performance: limit the number of distinct snapshot opponents sampled per hand.
    # Large pools can otherwise cause many tiny per-snapshot forward passes and tank SPS.
    league_rollout_snapshot_k: int
    # Self-play evaluation vs snapshot pool (stronger than bots; detects plateaus).
    # Requires `league_enabled=true` and `league_snapshot_pool>0`.
    selfplay_eval_enabled: bool
    selfplay_eval_every_updates: int
    selfplay_eval_hands: int
    selfplay_eval_subset: int
    selfplay_eval_deterministic: bool
    selfplay_eval_seat_swap: bool
    # If true, synchronizes CUDA at phase boundaries when timing the update loop.
    # Useful for diagnosing slowdowns, but adds overhead.
    timing_sync: bool
    # torch.compile() optimization for policy network (PyTorch 2.0+).
    compile_policy: bool
    # Automatic mixed precision (AMP) for faster training on modern GPUs.
    use_amp: bool
    # AMP dtype: "bfloat16" (recommended, no loss scaling needed) or "float16".
    amp_dtype: str
    # CUDA graphs for PPO update (experimental).
    # Captures the PPO minibatch update as a CUDA graph for reduced CPU overhead.
    # Requirements: minibatch_size must evenly divide (num_envs * rollout_steps).
    # Note: Currently not compatible with league_enabled=true due to conditional logic.
    use_cuda_graph: bool
    # If true, binary-search for the largest num_envs that fits in GPU memory before
    # training starts, then override num_envs/minibatch_size automatically.
    autotune: bool = False
    # Policy backbone: "lstm" (default) or "transformer".
    # Transformer = 4-layer encoder over a public-action token sequence,
    # carrying per-env (tokens, lengths) state packed into the existing
    # h tensor (see src/gpu_poker/policy_transformer.py). Most policy
    # hyperparameters (card_embed_dim, mlp_dim, torso_layers, lstm_hidden,
    # head_layers, head_dim) are LSTM-only and ignored by the transformer.
    policy_backbone: str = "lstm"
    # Transformer-specific arch knobs (used only when policy_backbone == "transformer").
    transformer_d_model: int = 256
    transformer_n_heads: int = 4
    transformer_n_layers: int = 4
    transformer_ffn_dim: int = 1024
    # PFSP priority function: "hard" (default, main-agent) or "var" (exploiter).
    league_pfsp_mode: str = "hard"
    # Exponent q for f_hard(p) = (1 - p)^q. AlphaStar default is 2.
    league_pfsp_q: float = 2.0
    # Sample scripted bots as training opponents (in addition to snapshots).
    # Default false: pure self-play vs the snapshot pool, bots stay eval-only.
    league_train_on_bots: bool = False
    # Randomized starting stack range in chips. When min < max, each env's
    # starting stack is sampled uniform in [min, max] once at training init.
    # Slumbot stakes (50/100 blinds): min=4000 (40 BB), max=20000 (200 BB)
    # gives a stack-depth-aware policy. Set min==max for fixed-stack training.
    starting_stack: int = 1000
    starting_stack_min: int = 1000
    starting_stack_max: int = 1000
    small_blind: int = 5
    big_blind: int = 10

    # =================== AlphaStar triad (G) ===================================
    # When `triad_enabled=true`, training runs three concurrent agents
    # (Main, Main Exploiter, League Exploiter) on partitioned slices of
    # `num_envs`, each with its own network + optimizer. See
    # `train/triad.py` for the controller. When false, behavior is
    # unchanged (single net, single optimizer, existing league logic).
    triad_enabled: bool = False
    # Per-role env partition (must sum to 1.0). Each role's slice trains
    # only its own network on its own RolloutBuffer.
    triad_main_frac: float = 0.6
    triad_me_frac: float = 0.2
    triad_le_frac: float = 0.2
    # Main snapshots every K updates and is demoted "main" -> "main_historical".
    triad_main_snapshot_every: int = 25
    # ME/LE snapshot only when WR vs their target(s) >= this threshold.
    triad_me_promote_wr: float = 0.70
    triad_le_promote_wr: float = 0.70
    # LE has a probabilistic reset on snapshot (paper says 25-50%).
    triad_le_reset_prob: float = 0.50
    # ME falls back to f_var sampling over historical mains when WR vs
    # current M is below this curriculum threshold (AlphaStar §2 trick).
    triad_me_curriculum_threshold: float = 0.30
    # Forced periodic ME reset: every N updates without a snapshot/reset,
    # copy current Main's weights into ME so ME gets a fresh shot at
    # finding an exploit. AlphaStar (Vinyals et al. 2019 §3 / Extended
    # Data Fig. 3) reinitializes main exploiters every 50-100 updates
    # regardless of whether they snapshotted, to escape local optima.
    # Without this, once Main catches up to ME, ME's WR collapses below
    # `triad_me_promote_wr` and ME can never snapshot again -- the
    # `reset_to_main` mutation only fires *on* snapshot. See
    # wiki/me-exploiter-stalling.md. 0 = disabled (matches pre-2026
    # behaviour); 50 = AlphaStar default.
    triad_me_force_reset_every: int = 0
    # Async eval cadence (updates) and hands per eval matchup.
    triad_eval_every_updates: int = 50
    triad_eval_hands: int = 1000
    # Main's opponent mix (must sum to 1.0).
    triad_main_mix_self: float = 0.35
    triad_main_mix_pfsp: float = 0.50
    triad_main_mix_forgotten: float = 0.15
    # Per-role snapshot pool capacity. ME=0 means ME snapshots are logged
    # for telemetry but never retained as sampleable opponents.
    triad_main_pool_cap: int = 16
    triad_le_pool_cap: int = 8
    triad_me_pool_cap: int = 0
    # Per-role hyperparameter overrides. `null` (None) means inherit the
    # global value of the corresponding field (lr / entropy_coef).
    triad_lr_main: float | None = None
    triad_lr_me: float | None = None
    triad_lr_le: float | None = None
    triad_entropy_me: float | None = None
    triad_entropy_le: float | None = None

    # ---- MLflow tracking -------------------------------------------------
    # Set False to skip MLflow entirely (smoke tests, offline runs, or
    # when the tracking server is down). All MLflow calls are also
    # wrapped in _safe_mlflow so a runtime failure won't crash training.
    mlflow_enabled: bool = True
    # Experiment to log under. Defaults to mlflow_config.DEFAULT_EXPERIMENT
    # when None so the constant stays the single source of truth.
    mlflow_experiment: str | None = None
    # Human-readable run name shown in the UI; auto-generated when None.
    mlflow_run_name: str | None = None
    # Neptune-style flat labels for the UI / search. Each becomes a
    # `label.<tag> = "true"` MLflow tag (searchable via
    # `tags."label.triad" = "true"`). Example: ["triad", "v100", "exp42"].
    mlflow_tags: list[str] | None = None
    # Free-form notes attached to the run (rendered as markdown in the
    # UI). Stored as the MLflow system tag `mlflow.note.content`.
    mlflow_notes: str | None = None
    # Extra arbitrary key-value tags (e.g. {"hypothesis": "raise more"}).
    # Merged in after the auto-tags so you can override.
    mlflow_extra_tags: dict[str, str] | None = None
    # When + which checkpoints to upload to MLflow's MinIO artifact store.
    # One of:
    #   "none"     -- never upload (default). .pt files stay on local disk.
    #   "final"    -- one upload at clean run-exit: the main net's
    #                 final weights as checkpoints/main/final.pt.
    #   "periodic" -- "final" PLUS every periodic save_every_env_steps tick.
    #   "all"      -- "periodic" PLUS every triad snapshot (M periodic,
    #                 ME/LE on WR-promotion) under checkpoints/snapshots/.
    # ~13MB per role per checkpoint at the default transformer size, so
    # "periodic"/"all" can push GBs over a long run -- stay on "final"
    # unless you need full provenance.
    mlflow_upload_checkpoints: str = "none"

    def __post_init__(self) -> None:
        # MLflow checkpoint-upload knob is validated regardless of triad_enabled
        # since it applies to both the single-net and triad paths.
        valid_upload_modes = {"none", "final", "periodic", "all"}
        if self.mlflow_upload_checkpoints not in valid_upload_modes:
            raise ValueError(
                f"mlflow_upload_checkpoints must be one of {sorted(valid_upload_modes)}, "
                f"got {self.mlflow_upload_checkpoints!r}"
            )

        # Validate triad config strictly at load time so the user sees a
        # clear error before any GPU is allocated (D11). Only checks the
        # triad block; the rest of the config is validated implicitly by
        # the type system + hydra schema.
        if not self.triad_enabled:
            return
        fracs_sum = self.triad_main_frac + self.triad_me_frac + self.triad_le_frac
        if abs(fracs_sum - 1.0) > 1e-6:
            raise ValueError(
                f"triad_main_frac + triad_me_frac + triad_le_frac must sum "
                f"to 1.0, got {fracs_sum:.6f}"
            )
        for role, frac in (
            ("main", self.triad_main_frac),
            ("me", self.triad_me_frac),
            ("le", self.triad_le_frac),
        ):
            if frac <= 0.0:
                raise ValueError(f"triad_{role}_frac must be > 0, got {frac}")
            slice_n = int(self.num_envs * frac)
            # A role needs enough envs to (a) form a non-degenerate rollout
            # and (b) produce a viable PPO minibatch. 64 is the empirical
            # floor below which PPO statistics get too noisy to learn.
            if slice_n < 64:
                raise ValueError(
                    f"triad_{role}_frac={frac} gives only {slice_n} envs "
                    f"(num_envs={self.num_envs}); need >= 64 per role"
                )
        mix_sum = (
            self.triad_main_mix_self + self.triad_main_mix_pfsp + self.triad_main_mix_forgotten
        )
        if abs(mix_sum - 1.0) > 1e-6:
            raise ValueError(
                f"triad_main_mix_self + triad_main_mix_pfsp + "
                f"triad_main_mix_forgotten must sum to 1.0, got {mix_sum:.6f}"
            )
        for name, val in (
            ("triad_main_pool_cap", self.triad_main_pool_cap),
            ("triad_le_pool_cap", self.triad_le_pool_cap),
            ("triad_me_pool_cap", self.triad_me_pool_cap),
        ):
            if val < 0:
                raise ValueError(f"{name} must be >= 0, got {val}")
        if not (0.0 < self.triad_me_promote_wr <= 1.0):
            raise ValueError(
                f"triad_me_promote_wr must be in (0, 1], got {self.triad_me_promote_wr}"
            )
        if not (0.0 < self.triad_le_promote_wr <= 1.0):
            raise ValueError(
                f"triad_le_promote_wr must be in (0, 1], got {self.triad_le_promote_wr}"
            )
        if not (0.0 <= self.triad_le_reset_prob <= 1.0):
            raise ValueError(
                f"triad_le_reset_prob must be in [0, 1], got {self.triad_le_reset_prob}"
            )


@dataclass(frozen=True)
class TrainResult:
    config: dict
    env_steps: int
    opt_steps: int
    elapsed_s: float
    overall_sps: float
    mean_loss: float
    approx_kl: float
    clipfrac: float
    value_loss: float
    policy_loss: float
    entropy: float
    invalid_rate: float
    eval: dict[str, float]


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
        self.scalars = torch.empty((t, n, scalar_dim), device=device, dtype=dtype)
        self.action_mask = torch.empty((t, n, c.NUM_ACTIONS), device=device, dtype=torch.bool)
        self.player_id = torch.empty((t, n), device=device, dtype=torch.int64)
        self.episode_id = torch.empty((t, n), device=device, dtype=torch.int32)
        self.reset_mask = torch.empty((t, n), device=device, dtype=torch.bool)
        self.learner_mask = torch.empty((t, n), device=device, dtype=torch.bool)

        self.action_type = torch.empty((t, n), device=device, dtype=torch.int64)
        # Discrete raise-bucket index in [0, NUM_RAISE_BUCKETS). int64 storage so
        # the replay-time Categorical.log_prob is exactly the same as at sample
        # time (no fp quantization to worry about — was a real bug for the old
        # continuous raise_frac field).
        self.raise_bucket = torch.empty((t, n), device=device, dtype=torch.int64)
        # Ratio-sensitive fields must stay in float32 regardless of the AMP
        # dtype used for scalar features. bf16 storage of logprob would quantize
        # the stored log_pi_old to a different point than the replay sees, so
        # exp(log_pi_new - log_pi_old) != 1 even with identical weights —
        # a silent bias in the PPO ratio. Same for value / reward / advantage
        # targets, and for the LSTM state snapshot used at replay.
        self.logprob = torch.empty((t, n), device=device, dtype=torch.float32)
        self.value = torch.empty((t, n), device=device, dtype=torch.float32)

        # Reward from the acting player's perspective.
        self.reward = torch.empty((t, n), device=device, dtype=torch.float32)
        # Terminal reward from P0 perspective (normalized); 0 on non-terminal steps.
        self.reward_p0 = torch.empty((t, n), device=device, dtype=torch.float32)
        self.done = torch.empty((t, n), device=device, dtype=torch.bool)

        # Acting-player LSTM state at action time (already reset as needed).
        self.h = torch.empty((t, n, hidden), device=device, dtype=torch.float32)
        self.c = torch.empty((t, n, hidden), device=device, dtype=torch.float32)


def _bucket_stats(raise_bucket_1d: torch.Tensor) -> dict[str, float]:
    """Per-bucket fraction of raise actions (raise_bucket: [K] int64 in [0, B))."""
    out: dict[str, float] = {}
    if raise_bucket_1d.numel() == 0:
        for b in range(NUM_RAISE_BUCKETS):
            out[f"bucket_{b}_pct"] = 0.0
        return out
    total = float(raise_bucket_1d.numel())
    for b in range(NUM_RAISE_BUCKETS):
        out[f"bucket_{b}_pct"] = float(raise_bucket_1d.eq(b).sum().item()) / total
    return out


def _action_stats(action_type: torch.Tensor, raise_bucket: torch.Tensor) -> dict[str, float]:
    # action_type: [T,N] int64, raise_bucket: [T,N] int64
    t, n = action_type.shape
    total = float(t * n)
    counts = {}
    for a in (c.ACTION_FOLD, c.ACTION_CHECK, c.ACTION_CALL, c.ACTION_RAISE):
        counts[a] = float(action_type.eq(a).sum().item())
    raise_mask = action_type.eq(c.ACTION_RAISE)
    rb = raise_bucket.squeeze(-1) if raise_bucket.ndim > 2 else raise_bucket
    rb_at_raise = rb[raise_mask]
    out = {
        "pct_fold": counts[c.ACTION_FOLD] / total,
        "pct_check": counts[c.ACTION_CHECK] / total,
        "pct_call": counts[c.ACTION_CALL] / total,
        "pct_raise": counts[c.ACTION_RAISE] / total,
    }
    out.update(_bucket_stats(rb_at_raise))
    return out


def _action_stats_masked(
    action_type: torch.Tensor, raise_bucket: torch.Tensor, mask: torch.Tensor
) -> dict[str, float]:
    # action_type: [T,N] int64, raise_bucket: [T,N] int64, mask: [T,N] bool
    mask = mask.to(dtype=torch.bool)
    total = float(mask.sum().item())
    if total <= 0:
        out = {
            "pct_fold": 0.0,
            "pct_check": 0.0,
            "pct_call": 0.0,
            "pct_raise": 0.0,
        }
        out.update(_bucket_stats(torch.empty(0, dtype=torch.int64)))
        return out
    at = action_type[mask]
    rb = raise_bucket.squeeze(-1) if raise_bucket.ndim > 2 else raise_bucket
    rb_all = rb[mask]
    counts = {}
    for a in (c.ACTION_FOLD, c.ACTION_CHECK, c.ACTION_CALL, c.ACTION_RAISE):
        counts[a] = float(at.eq(a).sum().item())
    raise_mask = at.eq(c.ACTION_RAISE)
    rb_at_raise = rb_all[raise_mask]
    out = {
        "pct_fold": counts[c.ACTION_FOLD] / total,
        "pct_check": counts[c.ACTION_CHECK] / total,
        "pct_call": counts[c.ACTION_CALL] / total,
        "pct_raise": counts[c.ACTION_RAISE] / total,
    }
    out.update(_bucket_stats(rb_at_raise))
    return out


def _action_stats_by_street(
    *,
    action_type: torch.Tensor,  # [T,N]
    raise_bucket: torch.Tensor,  # [T,N]
    scalars: torch.Tensor,  # [T,N,S]
    base_mask: torch.Tensor,  # [T,N] bool
) -> dict[str, dict[str, float]]:
    # scalars[...,7] is stage normalized by 5.0; map back to int stage.
    stage = (scalars[..., 7] * 5.0).round().to(dtype=torch.int64).clamp_(0, 5)
    out: dict[str, dict[str, float]] = {}
    for name, st in [
        ("preflop", c.STAGE_PREFLOP),
        ("flop", c.STAGE_FLOP),
        ("turn", c.STAGE_TURN),
        ("river", c.STAGE_RIVER),
    ]:
        m = base_mask & stage.eq(int(st))
        out[name] = _action_stats_masked(action_type, raise_bucket, m)
    return out


def _compute_gae(
    *,
    rewards: torch.Tensor,  # [T,B]
    dones: torch.Tensor,  # [T,B] bool
    values: torch.Tensor,  # [T,B]
    last_value: torch.Tensor,  # [B]
    player_id: torch.Tensor,  # [T,B] int, value perspective for each timestep
    last_player_id: torch.Tensor,  # [B] int, value perspective for last_value
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    t_steps, batch = rewards.shape
    advantages = torch.zeros((t_steps, batch), device=rewards.device, dtype=torch.float32)
    gae = torch.zeros((batch,), device=rewards.device, dtype=torch.float32)
    next_value = last_value
    for t in range(t_steps - 1, -1, -1):
        not_done = (~dones[t]).to(dtype=torch.float32)
        next_player = last_player_id if t == t_steps - 1 else player_id[t + 1]
        perspective_sign = torch.where(
            next_player.eq(player_id[t]),
            torch.ones((batch,), device=rewards.device, dtype=torch.float32),
            -torch.ones((batch,), device=rewards.device, dtype=torch.float32),
        )
        delta = rewards[t] + gamma * not_done * perspective_sign * next_value - values[t]
        gae = delta + gamma * gae_lambda * not_done * perspective_sign * gae
        advantages[t] = gae
        next_value = values[t]
    returns = advantages + values
    return advantages, returns


def _compute_mc_returns_p0(
    *,
    reward_p0: torch.Tensor,  # [T,N] float32 terminal-only (0 otherwise)
    dones: torch.Tensor,  # [T,N] bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Monte-Carlo episode returns from the P0 perspective.

    Used for league training where the opponent may be a different policy. Since env
    observations are active-player (include the acting player's private hole cards),
    we avoid requiring value targets on opponent-to-act states to prevent information leaks.

    Returns:
      returns_p0: [T,N] terminal return propagated backward within the rollout
      valid:      [T,N] True if the episode terminates within the rollout window
    """
    t_steps, n = reward_p0.shape
    returns = torch.zeros_like(reward_p0)
    valid = torch.zeros_like(dones)
    ret = torch.zeros((n,), device=reward_p0.device, dtype=torch.float32)
    has = torch.zeros((n,), device=reward_p0.device, dtype=torch.bool)
    for t in range(t_steps - 1, -1, -1):
        term = dones[t]
        ret = torch.where(term, reward_p0[t], ret)
        has = term | has
        returns[t] = ret
        valid[t] = has
    return returns, valid


def _push_transformer_token(
    h0: torch.Tensor,
    h1: torch.Tensor,
    *,
    player_id: torch.Tensor,
    action_type: torch.Tensor,
    raise_bucket: torch.Tensor,
    stage: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append one public-action token to BOTH seats' transformer state
    buffers (the action is public information). No-op for callers that
    aren't using the transformer backbone — those should branch on
    `cfg.policy_backbone` and skip this entirely.

    `raise_bucket` is zeroed for non-RAISE actions before the push so the
    history token for a fold/check/call carries no spurious bucket info.
    """
    bucket = torch.where(
        action_type.eq(c.ACTION_RAISE),
        raise_bucket.to(dtype=torch.int32),
        torch.zeros_like(raise_bucket, dtype=torch.int32),
    )
    pid_i32 = player_id.to(dtype=torch.int32)
    h0_new = push_action_token_packed(
        h0, player_id=pid_i32, action_type=action_type, raise_bucket=bucket, stage=stage
    )
    h1_new = push_action_token_packed(
        h1, player_id=pid_i32, action_type=action_type, raise_bucket=bucket, stage=stage
    )
    return h0_new, h1_new


def _stage_from_scalars(scalars: torch.Tensor) -> torch.Tensor:
    """Recover the int stage index (0..5) from the obs scalars snapshot.
    The observation kernel writes `scalars[:, 7] = state.stage / 5.0`.
    """
    return (scalars[..., 7] * 5.0).round().to(dtype=torch.int32)


def _select_player_state(
    *,
    player_id: torch.Tensor,
    h0: torch.Tensor,
    c0: torch.Tensor,
    h1: torch.Tensor,
    c1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # player_id: [N] in {0,1}; each h/c: [1,N,H]
    sel = player_id.view(1, -1, 1).to(dtype=torch.bool)
    h = torch.where(sel, h1, h0)
    c_ = torch.where(sel, c1, c0)
    return h, c_


def _scatter_player_state(
    *,
    player_id: torch.Tensor,
    h_new: torch.Tensor,
    c_new: torch.Tensor,
    h0: torch.Tensor,
    c0: torch.Tensor,
    h1: torch.Tensor,
    c1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Update only the acting player's slot for each env.
    is_p1 = player_id.to(dtype=torch.bool).view(1, -1, 1)
    h0 = torch.where(is_p1, h0, h_new)
    c0 = torch.where(is_p1, c0, c_new)
    h1 = torch.where(is_p1, h_new, h1)
    c1 = torch.where(is_p1, c_new, c1)
    return h0, c0, h1, c1


def run(cfg: TrainConfig) -> TrainResult:
    # nvmlInit() can raise NVMLError_LibRmVersionMismatch on hosts where the
    # NVIDIA driver and the libnvidia-ml.so userspace lib drifted out of sync
    # (common on rented infra). Treat that as "no NVML available" -- gpu_util
    # metrics will just be missing, training proceeds normally.
    use_nvml = _HAS_PYNVML and cfg.device.startswith("cuda")
    if use_nvml:
        try:
            pynvml.nvmlInit()
        except Exception as e:  # noqa: BLE001
            print(f"[warn] pynvml.nvmlInit() failed ({e!r}); disabling NVML metrics")
            use_nvml = False
    try:
        if cfg.triad_enabled:
            return _run_impl_triad(cfg)
        return _run_impl(cfg)
    finally:
        if use_nvml:
            with contextlib.suppress(Exception):
                pynvml.nvmlShutdown()


def _run_impl(cfg: TrainConfig) -> TrainResult:
    # Hydra changes CWD into the run directory, so repo-relative paths must be
    # normalized before any checkpoint IO.
    cfg_dict = None
    if not os.path.isabs(cfg.checkpoint_dir):
        cfg_dict = asdict(cfg)
        cfg_dict["checkpoint_dir"] = str(_REPO_ROOT / cfg.checkpoint_dir)
    if cfg.resume_from and not os.path.isabs(cfg.resume_from):
        if cfg_dict is None:
            cfg_dict = asdict(cfg)
        cfg_dict["resume_from"] = str(_REPO_ROOT / cfg.resume_from)
    if cfg_dict is not None:
        cfg = TrainConfig(**cfg_dict)

    torch.manual_seed(cfg.seed)
    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device(cfg.device)
    env = WarpPokerEnv(
        num_envs=cfg.num_envs,
        starting_stack=int(cfg.starting_stack),
        small_blind=int(cfg.small_blind),
        big_blind=int(cfg.big_blind),
        device=cfg.device,
    )
    # Randomized per-env starting stack: teaches the policy stack-depth-aware
    # play (40 BB push/fold all the way up to 200 BB 3-bet wars) instead of
    # overfitting to one specific depth. Fixed stack if min == max.
    if int(cfg.starting_stack_min) < int(cfg.starting_stack_max):
        per_env_stack = torch.randint(
            int(cfg.starting_stack_min),
            int(cfg.starting_stack_max) + 1,
            (cfg.num_envs,),
            device=device,
            dtype=torch.int32,
        )
        env.reset_with_config(starting_stack=per_env_stack)
        print(
            f"[env] randomized starting_stack in "
            f"[{int(cfg.starting_stack_min)}, {int(cfg.starting_stack_max)}] chips "
            f"({int(cfg.starting_stack_min) // int(cfg.big_blind)}-"
            f"{int(cfg.starting_stack_max) // int(cfg.big_blind)} BB)"
        )
    obs = env.reset()

    scalar_dim = int(obs["scalars"].shape[1])
    net = _make_policy(cfg, scalar_dim, device)

    if cfg.compile_policy:
        net = torch.compile(net, mode="reduce-overhead")  # type: ignore[assignment]

    # Use fused optimizer for better performance (single kernel for entire update).
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, fused=True)

    # Automatic mixed precision (AMP) setup.
    # BF16 is preferred (same exponent range as FP32, no loss scaling) but is only
    # natively supported on Ampere+ (sm_80+). "auto" picks bf16 if hardware supports
    # it, otherwise fp16 — important for V100 (sm_70) where bf16 has no hw acceleration.
    amp_dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    if cfg.amp_dtype == "auto":
        # Native bf16 requires compute capability >= 8.0 (Ampere). PyTorch's
        # is_bf16_supported() returns True even on V100 (sm_70) because it can be
        # emulated via fp32, but there's no perf benefit — and it can be slower
        # than fp16 since fp16 has Tensor Core acceleration on V100.
        cap = torch.cuda.get_device_capability() if cfg.device.startswith("cuda") else (0, 0)
        if cap >= (8, 0):
            amp_dtype = torch.bfloat16
            print(f"[amp] auto -> bfloat16 (sm_{cap[0]}{cap[1]}, native hardware support)")
        else:
            amp_dtype = torch.float16
            print(f"[amp] auto -> float16 (sm_{cap[0]}{cap[1]}, bf16 not hw-accelerated)")
    else:
        amp_dtype = amp_dtype_map.get(cfg.amp_dtype, torch.bfloat16)
    amp_enabled = cfg.use_amp and cfg.device.startswith("cuda")

    # GradScaler is REQUIRED for fp16 to prevent gradient underflow (Micikevicius
    # et al., "Mixed Precision Training", 2017). bf16 has the same exponent range
    # as fp32 so no scaling is needed — enabling the scaler for bf16 would be a
    # no-op pass-through but we explicitly disable it for clarity.
    scaler_enabled = amp_enabled and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler(device="cuda", enabled=scaler_enabled)
    if scaler_enabled:
        print("[amp] GradScaler enabled (fp16 needs loss scaling to avoid underflow)")

    # League (NFSP-lite): sample opponents per-env at hand boundaries.
    # Encoding:
    #   opp_kind: 0=self-play (opponent==learner), 1=bot, 2=snapshot
    #   opp_id:   unified-pool index for bot/snapshot opponents
    bot_names = train_bots.available_bot_names()
    opp_kind = torch.zeros((cfg.num_envs,), device=device, dtype=torch.int64)
    opp_id = torch.zeros((cfg.num_envs,), device=device, dtype=torch.int64)
    learner_seat = torch.zeros((cfg.num_envs,), device=device, dtype=torch.int64)

    pool = None
    opponent_metas: list[OpponentMeta] = []
    opponent_scores = torch.empty((0,), device=device, dtype=torch.float32)
    opponent_kinds = torch.empty((0,), device=device, dtype=torch.int64)
    opponent_filled = 0
    snapshot_models: dict[str, PolicyNet] = {}

    if cfg.league_enabled:
        pool_dir = Path(cfg.checkpoint_dir) / "league_pool"
        pool = UnifiedOpponentPool(
            pool_dir=pool_dir,
            bot_names=bot_names,
        )

        # In-memory opponent tracking
        max_opponents = len(bot_names) + cfg.league_snapshot_pool
        opponent_metas = pool.load_index()
        opponent_scores = torch.zeros(max_opponents, device=device)
        opponent_kinds = torch.zeros(
            max_opponents, device=device, dtype=torch.int64
        )  # 1=bot, 2=snapshot

        # Initialize scores from pool
        for i, meta in enumerate(opponent_metas[:max_opponents]):
            opponent_scores[i] = (
                float(meta.bb_per_hand)
                if meta.bb_per_hand is not None
                else float(cfg.league_bot_initial_score)
            )
            opponent_kinds[i] = 1 if meta.kind == "bot" else 2

        opponent_filled = min(len(opponent_metas), max_opponents)
    snapshot_adds = 0

    def resample_roles(mask: torch.Tensor) -> None:
        idx = torch.nonzero(mask.to(dtype=torch.bool), as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            return
        num = int(idx.numel())

        learner_seat[idx] = torch.randint(0, 2, (num,), device=device, dtype=torch.int64)
        u = torch.rand((num,), device=device, dtype=torch.float32)

        is_selfplay = u < float(cfg.league_eta_selfplay)
        want_opponent = ~is_selfplay

        opp_kind[idx[is_selfplay]] = 0
        opp_id[idx[is_selfplay]] = 0

        # Opponent from unified pool (PFSP-weighted)
        if want_opponent.any():
            if opponent_filled > 0:
                # Include main historical agents and exploiters. Scripted bots
                # are only included if `league_train_on_bots=true` (false by
                # default — bots are eval-only per the poker-AI SOTA recipes).
                valid_indices = [
                    i
                    for i, meta in enumerate(opponent_metas[:opponent_filled])
                    if meta.agent_role in {"main", "main_historical", "league_exploiter"}
                    or (meta.kind == "bot" and cfg.league_train_on_bots)
                ]
                snapshot_indices = [
                    i for i in valid_indices if opponent_metas[i].kind == "snapshot"
                ]
                max_rollout_snapshots = int(cfg.league_rollout_snapshot_k)
                if max_rollout_snapshots > 0 and len(snapshot_indices) > max_rollout_snapshots:
                    snap_t = torch.tensor(snapshot_indices, device=device, dtype=torch.int64)
                    snap_weights = pfsp_weights_from_bb_per_hand(
                        opponent_scores[snap_t],
                        temperature=cfg.league_pfsp_temperature,
                        epsilon=cfg.league_pfsp_epsilon,
                        mode=cfg.league_pfsp_mode,
                        q=cfg.league_pfsp_q,
                    )
                    snap_weights = snap_weights / snap_weights.sum().clamp_min(1e-12)
                    picked = torch.multinomial(
                        snap_weights,
                        num_samples=max_rollout_snapshots,
                        replacement=False,
                    )
                    picked_snapshots = set(snap_t[picked].tolist())
                    valid_indices = [
                        i
                        for i in valid_indices
                        if opponent_metas[i].kind == "bot" or i in picked_snapshots
                    ]

                if valid_indices:
                    indices_t = torch.tensor(valid_indices, device=device, dtype=torch.int64)
                    weights = pfsp_weights_from_bb_per_hand(
                        opponent_scores[indices_t],
                        temperature=cfg.league_pfsp_temperature,
                        epsilon=cfg.league_pfsp_epsilon,
                        mode=cfg.league_pfsp_mode,
                        q=cfg.league_pfsp_q,
                    )
                    weights = weights / weights.sum().clamp_min(1e-12)

                    n_samples = int(want_opponent.sum().item())
                    choice_idx = torch.multinomial(weights, num_samples=n_samples, replacement=True)
                    choice = indices_t[choice_idx]

                    opp_id[idx[want_opponent]] = choice
                    opp_kind[idx[want_opponent]] = opponent_kinds[choice]
                else:
                    # Fallback to bots/self-play if no specific roles found
                    opp_kind[idx[want_opponent]] = 0
                    opp_id[idx[want_opponent]] = 0
            else:
                # Fallback to self-play if pool is empty (shouldn't happen if bots exist)
                opp_kind[idx[want_opponent]] = 0
                opp_id[idx[want_opponent]] = 0

        # Update global state
        # Note: opp_kind/opp_id were updated in-place via slicing above

    def _pick_snapshot_eval_ids(*, subset: int, k_recent: int = 4) -> list[int]:
        # Filter for snapshots only
        snap_indices = [
            i for i, m in enumerate(opponent_metas[:opponent_filled]) if m.kind == "snapshot"
        ]

        if not snap_indices:
            return []

        subset = int(subset)
        if subset <= 0 or subset >= len(snap_indices):
            return snap_indices

        sorted_snaps = sorted(
            snap_indices,
            key=lambda i: (opponent_metas[i].env_steps or 0, opponent_metas[i].update or 0),
        )

        k_recent = int(min(k_recent, subset, len(snap_indices)))
        recent = sorted_snaps[-k_recent:]
        recent_set = set(recent)

        remaining = subset - k_recent
        if remaining <= 0:
            return recent

        cand = torch.tensor(
            [i for i in sorted_snaps if i not in recent_set],
            device=device,
            dtype=torch.int64,
        )
        if cand.numel() == 0:
            return recent

        r = min(remaining, int(cand.numel()))
        pick = cand[torch.randperm(int(cand.numel()), device=device)[:r]].tolist()
        return recent + pick

    if cfg.league_enabled:
        resample_roles(torch.ones((cfg.num_envs,), device=device, dtype=torch.bool))

    # Optional resume: loads model/optimizer and restores counters/RNG.
    update = 0
    ckpt_state = None
    if cfg.resume_from:
        ckpt_state = load_checkpoint(
            path=cfg.resume_from,
            model=net,
            optimizer=opt,
            device=device,
        )
        update = ckpt_state.update

    writer = None
    if _HAS_TB:
        tb_dir = os.environ.get("POKERGPU_TRAIN_TB_DIR", "train/tensorboard")
        writer = SummaryWriter(log_dir=tb_dir)

    buf = RolloutBuffer(
        t=cfg.rollout_steps,
        n=cfg.num_envs,
        scalar_dim=scalar_dim,
        hidden=net.lstm_hidden,
        device=device,
        dtype=amp_dtype if amp_enabled else torch.float32,
    )

    # Per-env per-player state: [1, N, H] - use buffer dtype for efficiency
    state_dtype = buf.dtype
    h0 = torch.zeros((1, cfg.num_envs, net.lstm_hidden), device=device, dtype=state_dtype)
    c0 = torch.zeros((1, cfg.num_envs, net.lstm_hidden), device=device, dtype=state_dtype)
    h1 = torch.zeros((1, cfg.num_envs, net.lstm_hidden), device=device, dtype=state_dtype)
    c1 = torch.zeros((1, cfg.num_envs, net.lstm_hidden), device=device, dtype=state_dtype)
    # Snapshot opponents are evaluated as memoryless during rollouts; feed constant zero state
    # to avoid per-step allocations.
    snap_h0 = torch.zeros_like(h0)
    snap_c0 = torch.zeros_like(c0)

    invalid_accum = torch.zeros((), device=device, dtype=torch.int64)

    t0 = time.perf_counter()
    # Accumulate on-device to avoid sync-heavy `.item()` inside the PPO loop.
    loss_sum_t = torch.zeros((), device=device, dtype=torch.float32)
    kl_sum_t = torch.zeros((), device=device, dtype=torch.float32)
    clipfrac_sum_t = torch.zeros((), device=device, dtype=torch.float32)
    value_loss_sum_t = torch.zeros((), device=device, dtype=torch.float32)
    policy_loss_sum_t = torch.zeros((), device=device, dtype=torch.float32)
    entropy_sum_t = torch.zeros((), device=device, dtype=torch.float32)
    opt_steps = 0
    env_steps = 0
    eval_metrics: dict[str, float] = {}
    eval_env: WarpPokerEnv | None = None

    def get_eval_env() -> WarpPokerEnv:
        nonlocal eval_env
        if eval_env is None:
            eval_env = WarpPokerEnv(num_envs=min(1024, cfg.num_envs), device=cfg.device)
        return eval_env

    # If we resumed, start counters from the checkpoint so:
    # - TensorBoard steps keep increasing (we use env_steps as the TB step axis)
    # - tqdm percent/progress starts where we left off
    if ckpt_state is not None:
        env_steps = int(ckpt_state.env_steps)
        opt_steps = int(ckpt_state.opt_steps)

    disable_pbar = os.environ.get("POKERGPU_TRAIN_TQDM", "1") == "0" or not sys.stderr.isatty()
    env_steps_per_rollout = int(cfg.num_envs) * int(cfg.rollout_steps)
    max_updates = cfg.num_updates if cfg.total_env_steps <= 0 else 0
    target_env_steps = cfg.total_env_steps if cfg.total_env_steps > 0 else 0
    pbar_total = (
        target_env_steps
        if target_env_steps > 0
        else (max_updates * env_steps_per_rollout if max_updates > 0 else 0)
    )
    pbar = tqdm(
        total=pbar_total if pbar_total > 0 else None,
        initial=env_steps if pbar_total > 0 else 0,
        disable=disable_pbar,
        desc="train",
        unit="env_step",
        unit_scale=True,
        unit_divisor=1000,
        mininterval=max(0.1, float(cfg.log_every_seconds)),
    )

    log_stride = cfg.log_every_env_steps if cfg.log_every_env_steps > 0 else env_steps_per_rollout
    next_log_env_steps = env_steps + log_stride
    next_log_time = time.perf_counter() + max(0.1, float(cfg.log_every_seconds))
    last_loss_u = float("nan")
    last_kl_u = float("nan")
    last_clip_u = float("nan")
    last_ent_u = float("nan")
    last_vloss_u = float("nan")
    last_ev = float("nan")

    def maybe_refresh(phase: str) -> None:
        nonlocal next_log_time
        if disable_pbar:
            return
        now = time.perf_counter()
        if now < next_log_time:
            return
        overall_sps_now = float(env_steps) / max(1e-9, now - t0)
        pbar.set_postfix(
            phase=phase,
            upd=update,
            all=f"{overall_sps_now / 1000:.0f}k",
            loss=f"{last_loss_u:.2f}" if last_loss_u == last_loss_u else "-",
            kl=f"{last_kl_u:.4f}" if last_kl_u == last_kl_u else "-",
            clip=f"{last_clip_u:.3f}" if last_clip_u == last_clip_u else "-",
            ent=f"{last_ent_u:.3f}" if last_ent_u == last_ent_u else "-",
            vloss=f"{last_vloss_u:.3f}" if last_vloss_u == last_vloss_u else "-",
            ev=f"{last_ev:.3f}" if last_ev == last_ev else "-",
            eval=f"{eval_metrics.get('bb_per_hand', 0.0):.3f}",
        )
        pbar.refresh()
        next_log_time = now + max(0.1, float(cfg.log_every_seconds))

    next_save_env_steps = (
        env_steps + cfg.save_every_env_steps if cfg.save_every_env_steps > 0 else 0
    )

    while True:
        if max_updates > 0 and update >= max_updates:
            break
        if target_env_steps > 0 and env_steps >= target_env_steps:
            break

        update_t0 = time.perf_counter()
        t_collect_s = 0.0
        t_opt_s = 0.0
        t_eval_s = 0.0
        t_league_eval_s = 0.0
        t_selfplay_eval_s = 0.0
        t_misc_s = 0.0
        mb_count = 0
        loss_sum_update_t = torch.zeros((), device=device, dtype=torch.float32)
        kl_sum_update_t = torch.zeros((), device=device, dtype=torch.float32)
        kl_abs_sum_update_t = torch.zeros((), device=device, dtype=torch.float32)
        clipfrac_sum_update_t = torch.zeros((), device=device, dtype=torch.float32)
        value_loss_sum_update_t = torch.zeros((), device=device, dtype=torch.float32)
        policy_loss_sum_update_t = torch.zeros((), device=device, dtype=torch.float32)
        entropy_sum_update_t = torch.zeros((), device=device, dtype=torch.float32)
        ev_rollout = 0.0
        # Diagnostics (counted over epoch 0 only, once per sample).
        diag_total_t = torch.zeros((), device=device, dtype=torch.float32)
        diag_old_nonfinite_t = torch.zeros((), device=device, dtype=torch.float32)
        diag_new_nonfinite_t = torch.zeros((), device=device, dtype=torch.float32)
        diag_log_ratio_nonfinite_t = torch.zeros((), device=device, dtype=torch.float32)
        diag_log_ratio_abs_sum_t = torch.zeros((), device=device, dtype=torch.float32)
        diag_log_ratio_abs_max_t = torch.zeros((), device=device, dtype=torch.float32)
        diag_old_lp_min_t = torch.full((), float("inf"), device=device, dtype=torch.float32)
        diag_old_lp_max_t = torch.full((), float("-inf"), device=device, dtype=torch.float32)
        diag_new_lp_min_t = torch.full((), float("inf"), device=device, dtype=torch.float32)
        diag_new_lp_max_t = torch.full((), float("-inf"), device=device, dtype=torch.float32)
        diag_log_ratio_min_t = torch.full((), float("inf"), device=device, dtype=torch.float32)
        diag_log_ratio_max_t = torch.full((), float("-inf"), device=device, dtype=torch.float32)

        if cfg.timing_sync:
            _sync(cfg.device)
        t_collect0 = time.perf_counter()
        with torch.no_grad():
            for t in range(cfg.rollout_steps):
                # If env just terminated on the previous step, reset both players.
                reset = obs["terminated"].to(dtype=torch.bool)
                if reset.any():
                    z = torch.zeros_like(h0)
                    h0 = torch.where(reset.view(1, -1, 1), z, h0)
                    c0 = torch.where(reset.view(1, -1, 1), z, c0)
                    h1 = torch.where(reset.view(1, -1, 1), z, h1)
                    c1 = torch.where(reset.view(1, -1, 1), z, c1)
                    if cfg.league_enabled:
                        resample_roles(reset)

                pid = obs["player_id"].to(dtype=torch.int64)
                ep_id = obs["episode_id"].to(dtype=torch.int32)

                h_batch, c_batch = _select_player_state(player_id=pid, h0=h0, c0=c0, h1=h1, c1=c1)

                # IMPORTANT: `obs` tensors are zero-copy views into env-owned buffers that
                # get overwritten in-place by `env.step()`. Snapshot the current observation
                # into the rollout buffer BEFORE stepping to keep (s_t, a_t, logp_t, mask_t)
                # aligned for PPO.
                buf.cards[t].copy_(obs["cards"])
                buf.scalars[t].copy_(obs["scalars"])
                buf.action_mask[t].copy_(obs["action_mask"])
                buf.player_id[t].copy_(pid)
                buf.episode_id[t].copy_(ep_id)
                buf.reset_mask[t].copy_(reset)
                buf.h[t].copy_(h_batch.squeeze(0))
                buf.c[t].copy_(c_batch.squeeze(0))

                if not cfg.league_enabled:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                        out = net.forward_step(
                            cards=obs["cards"],
                            scalars=obs["scalars"],
                            action_mask=obs["action_mask"],
                            h=h_batch,
                            c=c_batch,
                            terminated=None,
                            deterministic=False,
                        )

                    # Update the acting player's hidden state.
                    h0, c0, h1, c1 = _scatter_player_state(
                        player_id=pid,
                        h_new=out.h,
                        c_new=out.c,
                        h0=h0,
                        c0=c0,
                        h1=h1,
                        c1=c1,
                    )

                    amounts = raise_bucket_to_amount(
                        raise_bucket=out.raise_bucket,
                        pot_total=obs["pot_total"],
                        min_raise=obs["min_raise"],
                        max_raise=obs["max_raise"],
                    )
                    amounts = torch.where(
                        out.action_type.eq(c.ACTION_RAISE),
                        amounts,
                        torch.zeros_like(amounts),
                    )
                    next_obs = env.step(out.action_type.to(dtype=torch.int32), amounts)

                    if cfg.policy_backbone == "transformer":
                        stage_at_action = _stage_from_scalars(buf.scalars[t])
                        h0, h1 = _push_transformer_token(
                            h0,
                            h1,
                            player_id=pid,
                            action_type=out.action_type.to(dtype=torch.int32),
                            raise_bucket=out.raise_bucket,
                            stage=stage_at_action,
                        )

                    bb = next_obs["big_blind"].to(dtype=torch.float32).clamp_min(1.0)
                    # Reward is always returned as P0 reward; map to acting player perspective.
                    reward_act = torch.where(
                        pid.eq(0), next_obs["rewards"], -next_obs["rewards"]
                    ).to(dtype=torch.float32)
                    reward_act = reward_act / (bb * float(c.DEFAULT_STACK_BB))

                    buf.learner_mask[t].fill_(True)
                    buf.action_type[t].copy_(out.action_type)
                    buf.raise_bucket[t].copy_(out.raise_bucket)
                    buf.logprob[t].copy_(out.logprob)
                    buf.value[t].copy_(out.value)
                    buf.reward[t].copy_(reward_act)
                    buf.reward_p0[t].copy_(
                        torch.where(
                            next_obs["terminated"].to(dtype=torch.bool),
                            next_obs["rewards"].to(dtype=torch.float32)
                            / (bb * float(c.DEFAULT_STACK_BB)),
                            torch.zeros_like(reward_act),
                        )
                    )
                    buf.done[t].copy_(next_obs["terminated"])

                    invalid_accum = invalid_accum + next_obs["invalid_action"].sum().to(
                        dtype=torch.int64
                    )

                    obs = next_obs
                    maybe_refresh("collect")
                    continue

                # League-enabled: action selection depends on per-env opponent assignment.
                is_selfplay = opp_kind.eq(0)
                is_learner_actor = is_selfplay | pid.eq(learner_seat)
                buf.learner_mask[t].copy_(is_learner_actor)

                action_type = torch.full(
                    (cfg.num_envs,), c.ACTION_FOLD, device=device, dtype=torch.int32
                )
                # Per-step staging tensors for the chosen action + ratio-sensitive
                # log-prob / value (the rollout buffer keeps these in fp32 to
                # avoid mixed-precision quantization breaking the PPO ratio).
                raise_bucket = torch.zeros((cfg.num_envs,), device=device, dtype=torch.int64)
                logprob = torch.zeros((cfg.num_envs,), device=device, dtype=torch.float32)
                value = torch.zeros((cfg.num_envs,), device=device, dtype=torch.float32)

                if is_learner_actor.any():
                    idx_l = torch.nonzero(is_learner_actor, as_tuple=False).squeeze(-1)
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                        out_l = net.forward_step(
                            cards=obs["cards"][idx_l],
                            scalars=obs["scalars"][idx_l],
                            action_mask=obs["action_mask"][idx_l],
                            h=h_batch[:, idx_l, :],
                            c=c_batch[:, idx_l, :],
                            terminated=None,
                            deterministic=False,
                        )
                    action_type[idx_l] = out_l.action_type.to(dtype=torch.int32)
                    raise_bucket[idx_l] = out_l.raise_bucket.to(dtype=raise_bucket.dtype)
                    logprob[idx_l] = out_l.logprob.to(dtype=logprob.dtype)
                    value[idx_l] = out_l.value.to(dtype=value.dtype)

                    pid_l = pid[idx_l].to(dtype=torch.bool)
                    if (~pid_l).any():
                        idx0 = idx_l[~pid_l]
                        h0[:, idx0, :] = out_l.h[:, ~pid_l, :]
                        c0[:, idx0, :] = out_l.c[:, ~pid_l, :]
                    if pid_l.any():
                        idx1 = idx_l[pid_l]
                        h1[:, idx1, :] = out_l.h[:, pid_l, :]
                        c1[:, idx1, :] = out_l.c[:, pid_l, :]

                is_opponent_actor = (~is_learner_actor) & (~is_selfplay)
                bot_mask = is_opponent_actor & opp_kind.eq(1)
                if bot_mask.any():
                    for pool_idx in torch.unique(opp_id[bot_mask]).tolist():
                        pool_idx = int(pool_idx)
                        meta = opponent_metas[pool_idx]
                        if meta.kind != "bot":  # Should be guaranteed by opp_kind logic
                            continue
                        bot_name = meta.id

                        m = bot_mask & opp_id.eq(pool_idx)
                        if not m.any():
                            continue
                        idx_b = torch.nonzero(m, as_tuple=False).squeeze(-1)
                        bot_fn = train_bots.get_bot_fn(bot_name)
                        a_b, rf_b = bot_fn(
                            {
                                "action_mask": obs["action_mask"][idx_b],
                                "min_raise": obs["min_raise"][idx_b],
                                "max_raise": obs["max_raise"][idx_b],
                            }
                        )
                        action_type[idx_b] = a_b
                        # Bots return a continuous fraction in [0, 1]; coarse-map
                        # to a bucket index for the discrete action space. (This
                        # path is normally inactive — bots are eval-only when
                        # cfg.league_train_on_bots=false.)
                        bucket_b = (
                            (rf_b.to(dtype=torch.float32) * (NUM_RAISE_BUCKETS - 1))
                            .round()
                            .clamp_(0, NUM_RAISE_BUCKETS - 1)
                            .to(dtype=torch.int64)
                        )
                        raise_bucket[idx_b] = bucket_b

                snap_mask = is_opponent_actor & opp_kind.eq(2)
                if snap_mask.any():
                    for pool_idx in torch.unique(opp_id[snap_mask]).tolist():
                        pool_idx = int(pool_idx)
                        meta = opponent_metas[pool_idx]
                        if meta.kind != "snapshot":
                            continue
                        snap_path = os.path.join(cfg.checkpoint_dir, meta.id)

                        if snap_path not in snapshot_models:
                            snapshot_models[snap_path] = load_snapshot_model(
                                snap_path, device, cfg, scalar_dim
                            )

                        snap_net = snapshot_models[snap_path]

                        m = snap_mask & opp_id.eq(pool_idx)
                        if not m.any():
                            continue
                        idx_s = torch.nonzero(m, as_tuple=False).squeeze(-1)

                        with torch.autocast(
                            device_type="cuda", dtype=amp_dtype, enabled=amp_enabled
                        ):
                            out_s = snap_net.forward_step(
                                cards=obs["cards"][idx_s],
                                scalars=obs["scalars"][idx_s],
                                action_mask=obs["action_mask"][idx_s],
                                h=snap_h0[:, idx_s, :],
                                c=snap_c0[:, idx_s, :],
                                terminated=None,
                                deterministic=True,
                            )
                        action_type[idx_s] = out_s.action_type.to(dtype=torch.int32)
                        raise_bucket[idx_s] = out_s.raise_bucket.to(dtype=raise_bucket.dtype)

                amounts = raise_bucket_to_amount(
                    raise_bucket=raise_bucket,
                    pot_total=obs["pot_total"],
                    min_raise=obs["min_raise"],
                    max_raise=obs["max_raise"],
                )
                amounts = torch.where(
                    action_type.eq(c.ACTION_RAISE),
                    amounts,
                    torch.zeros_like(amounts),
                )
                next_obs = env.step(action_type, amounts)

                if cfg.policy_backbone == "transformer":
                    stage_at_action = _stage_from_scalars(buf.scalars[t])
                    h0, h1 = _push_transformer_token(
                        h0,
                        h1,
                        player_id=pid,
                        action_type=action_type,
                        raise_bucket=raise_bucket,
                        stage=stage_at_action,
                    )

                bb = next_obs["big_blind"].to(dtype=torch.float32).clamp_min(1.0)
                reward_p0 = next_obs["rewards"].to(dtype=torch.float32) / (
                    bb * float(c.DEFAULT_STACK_BB)
                )
                reward_p0 = torch.where(
                    next_obs["terminated"].to(dtype=torch.bool),
                    reward_p0,
                    torch.zeros_like(reward_p0),
                )

                buf.action_type[t].copy_(action_type.to(dtype=torch.int64))
                buf.raise_bucket[t].copy_(raise_bucket)
                buf.logprob[t].copy_(logprob)
                buf.value[t].copy_(value)
                buf.reward[t].zero_()
                buf.reward_p0[t].copy_(reward_p0)
                buf.done[t].copy_(next_obs["terminated"])

                invalid_accum = invalid_accum + next_obs["invalid_action"].sum().to(
                    dtype=torch.int64
                )

                obs = next_obs
                maybe_refresh("collect")

            # Bootstrap value at the end of rollout from the current acting player.
            if not cfg.league_enabled:
                pid_last = obs["player_id"].to(dtype=torch.int64)
                h_last, c_last = _select_player_state(
                    player_id=pid_last, h0=h0, c0=c0, h1=h1, c1=c1
                )
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                    out_last = net.forward_step(
                        cards=obs["cards"],
                        scalars=obs["scalars"],
                        action_mask=obs["action_mask"],
                        h=h_last,
                        c=c_last,
                        terminated=None,
                        deterministic=True,
                    )
                last_value = out_last.value
                last_player_id = pid_last
            else:
                # Bootstrap value path mirrors buf.value dtype (fp32) so the
                # advantage computation has no implicit downcast.
                last_value = torch.zeros((cfg.num_envs,), device=device, dtype=torch.float32)
                last_player_id = torch.zeros((cfg.num_envs,), device=device, dtype=torch.int64)

        # Advance progress as soon as we have collected the rollout, before PPO updates.
        env_steps += env_steps_per_rollout
        pbar.update(env_steps_per_rollout)
        if cfg.timing_sync:
            _sync(cfg.device)
        t_collect_s = time.perf_counter() - t_collect0

        # Determine whether we will log this update BEFORE running PPO.
        do_log = env_steps >= next_log_env_steps

        if cfg.timing_sync:
            _sync(cfg.device)
        t_opt0 = time.perf_counter()

        if not cfg.league_enabled:
            advantages, returns = _compute_gae(
                rewards=buf.reward,
                dones=buf.done,
                values=buf.value,
                last_value=last_value,
                player_id=buf.player_id,
                last_player_id=last_player_id,
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
            )

            adv = advantages.reshape(-1)
            adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-6)

            # Flatten rollout for minibatching.
            cards_f = buf.cards.reshape(-1, 7)
            scalars_f = buf.scalars.reshape(-1, scalar_dim)
            mask_f = buf.action_mask.reshape(-1, c.NUM_ACTIONS)
            action_type_f = buf.action_type.reshape(-1)
            raise_bucket_f = buf.raise_bucket.reshape(-1)
            old_logprob_f = buf.logprob.reshape(-1)
            returns_f = returns.reshape(-1)
            values_f = buf.value.reshape(-1)
            h_f = buf.h.reshape(-1, net.lstm_hidden)
            c_f = buf.c.reshape(-1, net.lstm_hidden)

            # Explained variance (critic health signal), computed on rollout-time values.
            var_v = returns_f.var(unbiased=False)
            if float(var_v.item()) < 1e-12:
                ev = torch.zeros((), device=device, dtype=torch.float32)
            else:
                ev = 1.0 - (returns_f - values_f).var(unbiased=False) / var_v
            ev_rollout = float(ev.detach().item())
            act_stats = _action_stats(buf.action_type, buf.raise_bucket)
            street_stats = _action_stats_by_street(
                action_type=buf.action_type,
                raise_bucket=buf.raise_bucket,
                scalars=buf.scalars,
                base_mask=torch.ones_like(buf.done, dtype=torch.bool),
            )

            total = int(cards_f.shape[0])
            train_mask_f = None
            total_effective = float(total)
        else:
            returns_p0, valid_p0 = _compute_mc_returns_p0(reward_p0=buf.reward_p0, dones=buf.done)
            returns_act = torch.where(buf.player_id.eq(0), returns_p0, -returns_p0)
            train_mask_f = (valid_p0 & buf.learner_mask).reshape(-1)
            if not bool(train_mask_f.any().item()):
                maybe_refresh("idle")
                update += 1
                continue

            # League training: keep the full rollout flattened and use `train_mask_f`
            # to zero-weight opponent steps and episodes that don't terminate within
            # the rollout window (no huge gathered tensors).
            cards_f = buf.cards.reshape(-1, 7)
            scalars_f = buf.scalars.reshape(-1, scalar_dim)
            mask_f = buf.action_mask.reshape(-1, c.NUM_ACTIONS)
            action_type_f = buf.action_type.reshape(-1)
            raise_bucket_f = buf.raise_bucket.reshape(-1)
            old_logprob_f = buf.logprob.reshape(-1)
            values_f = buf.value.reshape(-1)
            h_f = buf.h.reshape(-1, net.lstm_hidden)
            c_f = buf.c.reshape(-1, net.lstm_hidden)
            returns_f = returns_act.reshape(-1)

            w_all = train_mask_f.to(dtype=torch.float32)
            denom_all = w_all.sum().clamp_min(1.0)
            adv_raw = (returns_f - values_f).detach()
            adv_mean = (adv_raw * w_all).sum() / denom_all
            adv_var = ((adv_raw - adv_mean).pow(2) * w_all).sum() / denom_all
            adv_std = adv_var.sqrt().clamp_min(1e-6)
            adv = (adv_raw - adv_mean) / adv_std

            v_mean = (returns_f * w_all).sum() / denom_all
            var_v = ((returns_f - v_mean).pow(2) * w_all).sum() / denom_all
            if float(var_v.item()) < 1e-12:
                ev = torch.zeros((), device=device, dtype=torch.float32)
            else:
                resid_var = ((returns_f - values_f).pow(2) * w_all).sum() / denom_all
                ev = 1.0 - resid_var / var_v
            ev_rollout = float(ev.detach().item())
            act_stats = _action_stats_masked(buf.action_type, buf.raise_bucket, buf.learner_mask)
            street_stats = _action_stats_by_street(
                action_type=buf.action_type,
                raise_bucket=buf.raise_bucket,
                scalars=buf.scalars,
                base_mask=buf.learner_mask,
            )

            total = int(cards_f.shape[0])
            total_effective = float(denom_all.detach().item())

        # NOTE: `torch.randperm(total, device=cuda)` becomes very expensive at large totals
        # (e.g. millions of samples) and can look like the run is "stuck".
        # Instead of a full permutation, we use a cheap per-epoch offset and iterate
        # contiguous minibatches. This is sufficient for PPO diagnostics and avoids
        # multi-minute stalls on giant batches.

        # Iterate contiguous minibatches (views) to avoid per-minibatch index tensor
        # allocations and expensive advanced indexing gathers.
        mb_size = int(cfg.minibatch_size)
        num_mbs = (total + mb_size - 1) // max(1, mb_size)
        for _epoch in range(cfg.num_epochs):
            offset_mb = (_epoch * 104729) % max(1, num_mbs)  # 104729 is a prime
            for mb_i in range(num_mbs):
                j = (mb_i + offset_mb) % num_mbs
                start = j * mb_size
                end = min(total, start + mb_size)
                if end <= start:
                    continue
                sl = slice(start, end)
                mb_count += 1

                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                    eval_out = net.evaluate_step(
                        cards=cards_f[sl],
                        scalars=scalars_f[sl],
                        action_mask=mask_f[sl],
                        action_type=action_type_f[sl],
                        raise_bucket=raise_bucket_f[sl],
                        h=h_f[sl].unsqueeze(0),
                        c=c_f[sl].unsqueeze(0),
                        terminated=None,
                    )

                _assert_finite("old_logprob", old_logprob_f[sl])
                _assert_finite("new_logprob", eval_out.logprob)

                # Guard against exp overflow (which can otherwise turn PPO into NaNs/inf).
                log_ratio_raw = eval_out.logprob - old_logprob_f[sl]
                log_ratio = log_ratio_raw.clamp(min=-20.0, max=20.0)
                ratio = torch.exp(log_ratio)

                # Diagnostics: only compute when we are going to log, and keep everything
                # on-device to avoid per-minibatch `.item()` sync storms.
                if do_log and _epoch == 0:
                    diag_total_t = diag_total_t + torch.tensor(
                        float(end - start), device=device, dtype=torch.float32
                    )
                    old_lp = old_logprob_f[sl]
                    new_lp = eval_out.logprob
                    diag_old_nonfinite_t = diag_old_nonfinite_t + (
                        (~torch.isfinite(old_lp)).to(dtype=torch.float32).sum()
                    )
                    diag_new_nonfinite_t = diag_new_nonfinite_t + (
                        (~torch.isfinite(new_lp)).to(dtype=torch.float32).sum()
                    )
                    diag_log_ratio_nonfinite_t = diag_log_ratio_nonfinite_t + (
                        (~torch.isfinite(log_ratio_raw)).to(dtype=torch.float32).sum()
                    )
                    diag_old_lp_min_t = torch.minimum(diag_old_lp_min_t, old_lp.min())
                    diag_old_lp_max_t = torch.maximum(diag_old_lp_max_t, old_lp.max())
                    diag_new_lp_min_t = torch.minimum(diag_new_lp_min_t, new_lp.min())
                    diag_new_lp_max_t = torch.maximum(diag_new_lp_max_t, new_lp.max())
                    diag_log_ratio_min_t = torch.minimum(diag_log_ratio_min_t, log_ratio_raw.min())
                    diag_log_ratio_max_t = torch.maximum(diag_log_ratio_max_t, log_ratio_raw.max())
                    diag_log_ratio_abs_sum_t = diag_log_ratio_abs_sum_t + log_ratio_raw.abs().sum()
                    diag_log_ratio_abs_max_t = torch.maximum(
                        diag_log_ratio_abs_max_t, log_ratio_raw.abs().max()
                    )
                unclipped = ratio * adv[sl]
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv[sl]
                if cfg.league_enabled:
                    w = train_mask_f[sl].to(dtype=torch.float32)
                    denom = w.sum().clamp_min(1.0)
                    policy_loss = -(w * torch.minimum(unclipped, clipped)).sum() / denom
                    value_loss = 0.5 * (w * (eval_out.value - returns_f[sl]).pow(2)).sum() / denom
                    entropy_bonus = (w * eval_out.entropy).sum() / denom
                else:
                    policy_loss = -torch.minimum(unclipped, clipped).mean()
                    value_loss = 0.5 * (eval_out.value - returns_f[sl]).pow(2).mean()
                    entropy_bonus = eval_out.entropy.mean()

                # Note: this common PPO "approx KL" can be near-zero due to sign cancellation.
                # We also log an absolute variant for a more reliable magnitude signal.
                logp_diff = -log_ratio_raw  # old - new
                finite = torch.isfinite(logp_diff)
                if cfg.league_enabled:
                    m = train_mask_f[sl] & finite
                    count = m.to(dtype=torch.float32).sum().clamp_min(1.0)
                    diff = torch.where(m, logp_diff, torch.zeros_like(logp_diff))
                    approx_kl = diff.sum() / count
                    approx_kl_abs = diff.abs().sum() / count
                    clipped_m = (torch.abs(ratio - 1.0) > cfg.clip_eps) & train_mask_f[sl]
                    clipfrac = clipped_m.to(dtype=torch.float32).sum() / denom
                else:
                    count = finite.to(dtype=torch.float32).sum().clamp_min(1.0)
                    diff = torch.where(finite, logp_diff, torch.zeros_like(logp_diff))
                    approx_kl = diff.sum() / count
                    approx_kl_abs = diff.abs().sum() / count
                    clipfrac = (
                        (torch.abs(ratio - 1.0) > cfg.clip_eps).to(dtype=torch.float32).mean()
                    )

                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_bonus

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                # Unscale gradients before clipping so max_grad_norm is in the
                # original (unscaled) gradient space.
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=cfg.max_grad_norm)
                scaler.step(opt)
                scaler.update()
                opt_steps += 1

                loss_sum_t = loss_sum_t + loss.detach()
                kl_sum_t = kl_sum_t + approx_kl.detach()
                clipfrac_sum_t = clipfrac_sum_t + clipfrac.detach()
                value_loss_sum_t = value_loss_sum_t + value_loss.detach()
                policy_loss_sum_t = policy_loss_sum_t + policy_loss.detach()
                entropy_sum_t = entropy_sum_t + entropy_bonus.detach()

                loss_sum_update_t = loss_sum_update_t + loss.detach()
                kl_sum_update_t = kl_sum_update_t + approx_kl.detach()
                kl_abs_sum_update_t = kl_abs_sum_update_t + approx_kl_abs.detach()
                clipfrac_sum_update_t = clipfrac_sum_update_t + clipfrac.detach()
                value_loss_sum_update_t = value_loss_sum_update_t + value_loss.detach()
                policy_loss_sum_update_t = policy_loss_sum_update_t + policy_loss.detach()
                entropy_sum_update_t = entropy_sum_update_t + entropy_bonus.detach()
                maybe_refresh("update")

        if cfg.timing_sync:
            _sync(cfg.device)
        t_opt_s = time.perf_counter() - t_opt0

        do_log = env_steps >= next_log_env_steps
        # Per-update means (avoid div-by-zero in small test configs).
        mb_den = max(1, mb_count)
        denom_t = torch.tensor(float(mb_den), device=device, dtype=torch.float32)
        loss_u = float((loss_sum_update_t / denom_t).item())
        kl_u = float((kl_sum_update_t / denom_t).item())
        kl_abs_u = float((kl_abs_sum_update_t / denom_t).item())
        clip_u = float((clipfrac_sum_update_t / denom_t).item())
        vloss_u = float((value_loss_sum_update_t / denom_t).item())
        ploss_u = float((policy_loss_sum_update_t / denom_t).item())
        ent_u = float((entropy_sum_update_t / denom_t).item())
        last_loss_u = loss_u
        last_kl_u = kl_u
        last_clip_u = clip_u
        last_ent_u = ent_u
        last_vloss_u = vloss_u
        last_ev = ev_rollout

        if do_log:
            next_log_env_steps += (
                cfg.log_every_env_steps if cfg.log_every_env_steps > 0 else env_steps_per_rollout
            )

        if writer is not None and do_log:
            if _HAS_PYNVML and cfg.device.startswith("cuda"):
                try:
                    dev_idx = int(cfg.device.split(":")[-1])
                    handle = pynvml.nvmlDeviceGetHandleByIndex(dev_idx)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    writer.add_scalar("system/gpu_util_pct", float(util.gpu), env_steps)
                    writer.add_scalar(
                        "system/gpu_mem_pct", float(100.0 * mem.used / mem.total), env_steps
                    )
                except Exception:
                    # Non-fatal; driver issues, etc.
                    pass

            overall_sps_now = float(env_steps) / max(1e-9, time.perf_counter() - t0)
            writer.add_scalar("train/overall_sps", overall_sps_now, env_steps)
            writer.add_scalar("train/loss", loss_u, env_steps)
            writer.add_scalar("train/approx_kl", kl_u, env_steps)
            writer.add_scalar("train/approx_kl_abs", kl_abs_u, env_steps)
            writer.add_scalar("train/clipfrac", clip_u, env_steps)
            writer.add_scalar("train/value_loss", vloss_u, env_steps)
            writer.add_scalar("train/policy_loss", ploss_u, env_steps)
            writer.add_scalar("train/entropy", ent_u, env_steps)
            writer.add_scalar("train/explained_variance", ev_rollout, env_steps)
            # PPO loop diagnostics (helps explain sudden t_opt spikes).
            writer.add_scalar("debug/ppo_total_samples", float(total_effective), env_steps)
            writer.add_scalar("debug/ppo_mb_count", float(mb_count), env_steps)
            writer.add_scalar("debug/ppo_minibatch_size", float(cfg.minibatch_size), env_steps)
            writer.add_scalar(
                "debug/ppo_samples_per_rollout",
                float(env_steps_per_rollout),
                env_steps,
            )
            writer.add_scalar(
                "debug/ppo_total_frac_of_rollout",
                float(total_effective) / float(max(1, env_steps_per_rollout)),
                env_steps,
            )
            if cfg.league_enabled and opponent_filled > 0:
                w = pfsp_weights_from_bb_per_hand(
                    opponent_scores[:opponent_filled],
                    temperature=cfg.league_pfsp_temperature,
                    epsilon=cfg.league_pfsp_epsilon,
                    mode=cfg.league_pfsp_mode,
                    q=cfg.league_pfsp_q,
                )
                w = w / w.sum().clamp_min(1e-12)
                writer.add_scalar("league/pfsp_weight_min", float(w.min().item()), env_steps)
                writer.add_scalar("league/pfsp_weight_max", float(w.max().item()), env_steps)
                ent = -(w * (w.clamp_min(1e-12).log())).sum()
                writer.add_scalar("league/pfsp_weight_entropy", float(ent.item()), env_steps)
                ess = 1.0 / float((w * w).sum().clamp_min(1e-12).item())
                writer.add_scalar("league/pfsp_weight_ess", float(ess), env_steps)

                # Log individual opponent weights
                for i, meta in enumerate(opponent_metas[:opponent_filled]):
                    weight = float(w[i].item())
                    if meta.kind == "bot":
                        name = f"bot_{meta.id}"
                    else:
                        # Sanitize snapshot path for a clean tag name
                        p = Path(meta.id)
                        name = f"snap_{p.stem}"
                    writer.add_scalar(f"league_weights/{name}", weight, env_steps)

                # Log per-opponent-type breakdown
                bot_mask = opponent_kinds[:opponent_filled] == 1
                snap_mask = opponent_kinds[:opponent_filled] == 2

                if bot_mask.any():
                    bot_weight_sum = w[bot_mask].sum()
                    writer.add_scalar("league/bot_weight_total", float(bot_weight_sum), env_steps)

                if snap_mask.any():
                    snap_weight_sum = w[snap_mask].sum()
                    writer.add_scalar(
                        "league/snapshot_weight_total", float(snap_weight_sum), env_steps
                    )

                # Rollout mixture composition (cheap diagnostic for throughput changes).
                writer.add_scalar(
                    "league/rollout_snap_frac",
                    float(opp_kind.eq(2).to(dtype=torch.float32).mean().item()),
                    env_steps,
                )
                writer.add_scalar(
                    "league/rollout_bot_frac",
                    float(opp_kind.eq(1).to(dtype=torch.float32).mean().item()),
                    env_steps,
                )
                writer.add_scalar(
                    "league/rollout_selfplay_frac",
                    float(opp_kind.eq(0).to(dtype=torch.float32).mean().item()),
                    env_steps,
                )
            writer.add_scalar("actions/pct_fold", act_stats["pct_fold"], env_steps)
            writer.add_scalar("actions/pct_check", act_stats["pct_check"], env_steps)
            writer.add_scalar("actions/pct_call", act_stats["pct_call"], env_steps)
            writer.add_scalar("actions/pct_raise", act_stats["pct_raise"], env_steps)
            for b in range(NUM_RAISE_BUCKETS):
                key = f"bucket_{b}_pct"
                writer.add_scalar(f"actions/raise_{key}", act_stats[key], env_steps)
            for street, st in street_stats.items():
                writer.add_scalar(f"actions/{street}/pct_fold", st["pct_fold"], env_steps)
                writer.add_scalar(f"actions/{street}/pct_check", st["pct_check"], env_steps)
                writer.add_scalar(f"actions/{street}/pct_call", st["pct_call"], env_steps)
                writer.add_scalar(f"actions/{street}/pct_raise", st["pct_raise"], env_steps)
                for b in range(NUM_RAISE_BUCKETS):
                    key = f"bucket_{b}_pct"
                    writer.add_scalar(f"actions/{street}/raise_{key}", st[key], env_steps)
            writer.add_scalar(
                "train/invalid_rate",
                float(invalid_accum.item()) / max(1, env_steps),
                env_steps,
            )
            diag_total = float(diag_total_t.item())
            if diag_total > 0:
                writer.add_scalar(
                    "debug/logprob_old_nonfinite_rate",
                    float((diag_old_nonfinite_t / diag_total_t.clamp_min(1.0)).item()),
                    env_steps,
                )
                writer.add_scalar(
                    "debug/logprob_new_nonfinite_rate",
                    float((diag_new_nonfinite_t / diag_total_t.clamp_min(1.0)).item()),
                    env_steps,
                )
                writer.add_scalar(
                    "debug/log_ratio_nonfinite_rate",
                    float((diag_log_ratio_nonfinite_t / diag_total_t.clamp_min(1.0)).item()),
                    env_steps,
                )
                writer.add_scalar(
                    "debug/log_ratio_abs_mean",
                    float((diag_log_ratio_abs_sum_t / diag_total_t.clamp_min(1.0)).item()),
                    env_steps,
                )
                writer.add_scalar(
                    "debug/log_ratio_abs_max",
                    float(diag_log_ratio_abs_max_t.item()),
                    env_steps,
                )
                writer.add_scalar(
                    "debug/logprob_old_min", float(diag_old_lp_min_t.item()), env_steps
                )
                writer.add_scalar(
                    "debug/logprob_old_max", float(diag_old_lp_max_t.item()), env_steps
                )
                writer.add_scalar(
                    "debug/logprob_new_min", float(diag_new_lp_min_t.item()), env_steps
                )
                writer.add_scalar(
                    "debug/logprob_new_max", float(diag_new_lp_max_t.item()), env_steps
                )
                writer.add_scalar(
                    "debug/log_ratio_min", float(diag_log_ratio_min_t.item()), env_steps
                )
                writer.add_scalar(
                    "debug/log_ratio_max", float(diag_log_ratio_max_t.item()), env_steps
                )

        # Periodic checkpointing by env steps.
        if next_save_env_steps > 0 and env_steps >= next_save_env_steps:
            if cfg.timing_sync:
                _sync(cfg.device)
            t_misc0 = time.perf_counter()
            ckpt_path = Path(cfg.checkpoint_dir) / f"ckpt_env{env_steps}_upd{update}.pt"
            save_checkpoint(
                path=ckpt_path,
                model=net,
                optimizer=opt,
                state=CheckpointState(env_steps=env_steps, opt_steps=opt_steps, update=update),
            )
            next_save_env_steps += cfg.save_every_env_steps
            if cfg.timing_sync:
                _sync(cfg.device)
            t_misc_s += time.perf_counter() - t_misc0

        if cfg.eval_every > 0 and (update + 1) % cfg.eval_every == 0:
            if cfg.timing_sync:
                _sync(cfg.device)
            t_eval0 = time.perf_counter()
            eval_env = get_eval_env()
            all_eval: dict[str, dict[str, float]] = {}
            for bot_name in train_bots.available_bot_names():
                # For a couple of key opponents, also collect per-street action diagnostics.
                track = bot_name in {"nit", "random_aggressive"}
                if track:
                    m0 = eval_vs_bot(
                        env=eval_env,
                        policy=net,
                        bot_name=bot_name,
                        target_hands=cfg.eval_hands,
                        max_steps=cfg.eval_max_steps,
                        learner_seat=0,
                        deterministic=False,
                        track_actions=True,
                    )
                    m1 = eval_vs_bot(
                        env=eval_env,
                        policy=net,
                        bot_name=bot_name,
                        target_hands=cfg.eval_hands,
                        max_steps=cfg.eval_max_steps,
                        learner_seat=1,
                        deterministic=False,
                        track_actions=True,
                    )
                    bb_avg = 0.5 * (float(m0["bb_per_hand"]) + float(m1["bb_per_hand"]))
                    hands_total = int(m0["hands"]) + int(m1["hands"])
                    m = {
                        "bb_per_hand": bb_avg,
                        "bb_per_100": bb_avg * 100.0,
                        "hands": float(hands_total),
                        "bb_per_hand_p0": float(m0["bb_per_hand"]),
                        "bb_per_hand_p1": float(m1["bb_per_hand"]),
                    }
                    # Prefer seat-0 action stats as a stable diagnostic view.
                    for k, v in m0.items():
                        if k.startswith("actions/"):
                            m[k] = float(v)
                else:
                    m = eval_vs_bot_seat_swap(
                        env=eval_env,
                        policy=net,
                        bot_name=bot_name,
                        target_hands=cfg.eval_hands,
                        max_steps=cfg.eval_max_steps,
                        deterministic=False,
                    )
                all_eval[bot_name] = m
                if writer is not None:
                    writer.add_scalar(f"eval/{bot_name}_bb_per_hand", m["bb_per_hand"], env_steps)
                    writer.add_scalar(f"eval/{bot_name}_bb_per_100", m["bb_per_100"], env_steps)
                    writer.add_scalar(f"eval/{bot_name}_hands", m["hands"], env_steps)
                    writer.add_scalar(
                        f"eval/{bot_name}_bb_per_hand_p0", m["bb_per_hand_p0"], env_steps
                    )
                    writer.add_scalar(
                        f"eval/{bot_name}_bb_per_hand_p1", m["bb_per_hand_p1"], env_steps
                    )
                    if track:
                        for street in ("preflop", "flop", "turn", "river"):
                            for stat in (
                                "pct_fold",
                                "pct_check",
                                "pct_call",
                                "pct_raise",
                                "raise_frac_mean",
                                "raise_frac_min",
                                "raise_frac_max",
                            ):
                                tag = f"actions/{street}/{stat}"
                                key = f"actions/{street}/{stat}"
                                if key in m:
                                    writer.add_scalar(
                                        f"eval_actions/{bot_name}/{tag}", m[key], env_steps
                                    )

            # Keep the trainer's "headline" eval metric stable for progress bars and
            # `results.json` by using calling_station when present.
            eval_metrics = all_eval.get("calling_station", next(iter(all_eval.values()), {}))

            # Stronger eval: current policy vs a snapshot-mixture opponent.
            snap_indices = [
                i for i, m in enumerate(opponent_metas[:opponent_filled]) if m.kind == "snapshot"
            ]
            if cfg.league_enabled and snap_indices and writer is not None:
                s_scores = opponent_scores[snap_indices]
                w = pfsp_weights_from_bb_per_hand(
                    s_scores,
                    temperature=cfg.league_pfsp_temperature,
                    epsilon=cfg.league_pfsp_epsilon,
                    mode=cfg.league_pfsp_mode,
                    q=cfg.league_pfsp_q,
                )

                snap_policies = []
                for i in snap_indices:
                    path = os.path.join(cfg.checkpoint_dir, opponent_metas[i].id)
                    if path not in snapshot_models:
                        snapshot_models[path] = load_snapshot_model(path, device, cfg, scalar_dim)
                    snap_policies.append(snapshot_models[path])

                mix = eval_vs_snapshot_mix_seat_swap(
                    env=eval_env,
                    policy=net,
                    snapshots=snap_policies,
                    snap_weights=w,
                    target_hands=cfg.eval_hands,
                    max_steps=cfg.eval_max_steps,
                    deterministic=False,
                )
                writer.add_scalar("eval/pool_mix_bb_per_hand", mix["bb_per_hand"], env_steps)
                writer.add_scalar("eval/pool_mix_bb_per_100", mix["bb_per_100"], env_steps)
                writer.add_scalar("eval/pool_mix_hands", mix["hands"], env_steps)
                writer.add_scalar("eval/pool_mix_bb_per_hand_p0", mix["bb_per_hand_p0"], env_steps)
                writer.add_scalar("eval/pool_mix_bb_per_hand_p1", mix["bb_per_hand_p1"], env_steps)
            if cfg.timing_sync:
                _sync(cfg.device)
            t_eval_s = time.perf_counter() - t_eval0

        # League: snapshot creation (disk-backed + in-memory ring).
        if (
            cfg.league_enabled
            and pool is not None
            and cfg.league_snapshot_every_updates > 0
            and (update + 1) % int(cfg.league_snapshot_every_updates) == 0
        ):
            if cfg.timing_sync:
                _sync(cfg.device)
            # Demote previous main agent to historical
            current_main = next((o for o in opponent_metas if o.agent_role == "main"), None)
            if current_main is not None:
                pool.update_role(opponent_id=current_main.id, new_role="main_historical")

            meta = pool.add_snapshot(
                model=net, env_steps=env_steps, update=update, agent_role="main"
            )

            # Update in-memory pool using ring buffer logic for snapshots
            num_bots = len(bot_names)
            snap_capacity = int(cfg.league_snapshot_pool)

            if snap_capacity > 0:
                slot = snapshot_adds % snap_capacity
                idx = num_bots + slot

                # If we are replacing an existing snapshot, remove its model from cache
                if idx < len(opponent_metas):
                    old_id = opponent_metas[idx].id
                    if old_id in snapshot_models:
                        del snapshot_models[old_id]

                # Update/Append meta
                if idx < len(opponent_metas):
                    opponent_metas[idx] = meta
                else:
                    opponent_metas.append(meta)

                # Update tensors
                if idx < max_opponents:
                    opponent_scores[idx] = 0.0  # Neutral start
                    opponent_kinds[idx] = 2
                    opponent_filled = max(opponent_filled, idx + 1)

                # Cache new model
                snapshot_models[meta.id] = copy.deepcopy(net)
                snapshot_models[meta.id].eval()

            snapshot_adds += 1

        # League: periodic eval vs unified pool (for PFSP sampling weights).
        if (
            cfg.league_enabled
            and opponent_filled > 0
            and cfg.league_eval_every_updates > 0
            and (update + 1) % int(cfg.league_eval_every_updates) == 0
        ):
            was_training = net.training
            net.eval()
            try:
                if cfg.timing_sync:
                    _sync(cfg.device)
                t_league0 = time.perf_counter()
                eval_env = get_eval_env()

                # 1. Evaluate Bots
                if cfg.league_eval_bots and (update + 1) % int(cfg.league_eval_bots_every) == 0:
                    for i, meta in enumerate(opponent_metas[:opponent_filled]):
                        if meta.kind != "bot":
                            continue

                        m = eval_vs_bot(
                            env=eval_env,
                            policy=net,
                            bot_name=meta.id,
                            target_hands=cfg.league_eval_hands,
                            max_steps=cfg.eval_max_steps,
                            deterministic=bool(cfg.league_eval_deterministic),
                        )
                        bb = float(m["bb_per_hand"])
                        opponent_scores[i] = bb

                        if pool is not None:
                            pool.update_eval(
                                opponent_id=meta.id,
                                bb_per_hand=bb,
                                hands=int(m["hands"]),
                            )
                        if writer is not None:
                            writer.add_scalar(f"league/bot_{meta.id}_bb_per_hand", bb, env_steps)

                # 2. Evaluate Snapshots
                # Filter for snapshots only
                snap_indices = [
                    i
                    for i, m in enumerate(opponent_metas[:opponent_filled])
                    if m.kind == "snapshot"
                ]

                if snap_indices:
                    subset = int(cfg.league_eval_subset)
                    eval_ids: list[int] = []

                    if subset <= 0 or subset >= len(snap_indices):
                        eval_ids = snap_indices
                    else:
                        # Pick most recent snapshots by env_steps.
                        # We need to sort snap_indices by the env_steps of the meta.
                        # Use metadata; snapshot_env_steps no longer exists.
                        sorted_snaps = sorted(
                            snap_indices,
                            key=lambda i: (
                                opponent_metas[i].env_steps or 0,
                                opponent_metas[i].update or 0,
                            ),
                        )

                        k_recent = min(4, subset)
                        recent = sorted_snaps[-k_recent:]
                        recent_set = set(recent)

                        remaining = subset - k_recent
                        if remaining > 0:
                            cand = [i for i in sorted_snaps if i not in recent_set]
                            if cand:
                                r = min(remaining, len(cand))
                                perm = torch.randperm(len(cand), device=device)[:r].tolist()
                                pick = [cand[p] for p in perm]
                                eval_ids = recent + pick
                            else:
                                eval_ids = recent
                        else:
                            eval_ids = recent

                    det = bool(cfg.league_eval_deterministic)
                    for i in eval_ids:
                        meta = opponent_metas[i]
                        snap_path = os.path.join(cfg.checkpoint_dir, meta.id)

                        if snap_path not in snapshot_models:
                            snapshot_models[snap_path] = load_snapshot_model(
                                snap_path, device, cfg, scalar_dim
                            )

                        snap_net = snapshot_models[snap_path]

                        m = eval_vs_policy(
                            env=eval_env,
                            policy_p0=net,
                            policy_p1=snap_net,
                            target_hands=cfg.league_eval_hands,
                            max_steps=cfg.eval_max_steps,
                            deterministic=det,
                        )
                        bb_a = float(m["bb_per_hand"])  # learner as P0
                        hands_a = int(m["hands"])

                        bb_b_learn = 0.0
                        hands_b = 0
                        if bool(cfg.league_eval_seat_swap):
                            m2 = eval_vs_policy(
                                env=eval_env,
                                policy_p0=snap_net,
                                policy_p1=net,
                                target_hands=cfg.league_eval_hands,
                                max_steps=cfg.eval_max_steps,
                                deterministic=det,
                            )
                            # m2 is snapshot as P0; learner is P1, so learner bb/hand is -P0.
                            bb_b_learn = -float(m2["bb_per_hand"])
                            hands_b = int(m2["hands"])

                        score = (
                            0.5 * (bb_a + bb_b_learn) if bool(cfg.league_eval_seat_swap) else bb_a
                        )
                        opponent_scores[i] = float(score)

                        if pool is not None:
                            if bool(cfg.league_eval_seat_swap):
                                total_hands = hands_a + hands_b
                            else:
                                total_hands = hands_a
                            pool.update_eval(
                                opponent_id=meta.id,
                                bb_per_hand=float(score),
                                hands=int(total_hands),
                                bb_per_hand_p0=bb_a,
                                bb_per_hand_p1=(
                                    bb_b_learn if bool(cfg.league_eval_seat_swap) else None
                                ),
                            )

                if writer is not None and snap_indices:
                    # Log stats for snapshots specifically
                    s_vals = opponent_scores[snap_indices]
                    writer.add_scalar("league/snapshots_count", len(snap_indices), env_steps)
                    writer.add_scalar(
                        "league/snapshot_bb_per_hand_mean", float(s_vals.mean().item()), env_steps
                    )
                    writer.add_scalar(
                        "league/snapshot_bb_per_hand_min", float(s_vals.min().item()), env_steps
                    )
                    writer.add_scalar(
                        "league/snapshot_bb_per_hand_max", float(s_vals.max().item()), env_steps
                    )

                if cfg.timing_sync:
                    _sync(cfg.device)
                t_league_eval_s = time.perf_counter() - t_league0
            finally:
                if was_training:
                    net.train()

        # Self-play eval vs snapshot pool (stronger than bots).
        has_snapshots = any(m.kind == "snapshot" for m in opponent_metas[:opponent_filled])
        if (
            cfg.selfplay_eval_enabled
            and cfg.league_enabled
            and has_snapshots
            and cfg.selfplay_eval_every_updates > 0
            and (update + 1) % int(cfg.selfplay_eval_every_updates) == 0
        ):
            was_training = net.training
            try:
                if cfg.timing_sync:
                    _sync(cfg.device)
                t_sp0 = time.perf_counter()
                net.eval()

                eval_env = get_eval_env()
                det = bool(cfg.selfplay_eval_deterministic)
                seat_swap = bool(cfg.selfplay_eval_seat_swap)
                eval_ids = _pick_snapshot_eval_ids(subset=int(cfg.selfplay_eval_subset))

                scores: list[float] = []
                hands_total: list[int] = []
                for i in eval_ids:
                    meta = opponent_metas[i]
                    snap_path = os.path.join(cfg.checkpoint_dir, meta.id)

                    if snap_path not in snapshot_models:
                        snapshot_models[snap_path] = load_snapshot_model(
                            snap_path, device, cfg, scalar_dim
                        )

                    snap_net = snapshot_models[snap_path]
                    snap_net.eval()

                    m = eval_vs_policy(
                        env=eval_env,
                        policy_p0=net,
                        policy_p1=snap_net,
                        target_hands=int(cfg.selfplay_eval_hands),
                        max_steps=cfg.eval_max_steps,
                        deterministic=det,
                    )
                    bb_a = float(m["bb_per_hand"])
                    hands_a = int(m["hands"])

                    bb_b_learn = 0.0
                    hands_b = 0
                    if seat_swap:
                        m2 = eval_vs_policy(
                            env=eval_env,
                            policy_p0=snap_net,
                            policy_p1=net,
                            target_hands=int(cfg.selfplay_eval_hands),
                            max_steps=cfg.eval_max_steps,
                            deterministic=det,
                        )
                        bb_b_learn = -float(m2["bb_per_hand"])
                        hands_b = int(m2["hands"])

                    score = 0.5 * (bb_a + bb_b_learn) if seat_swap else bb_a
                    scores.append(float(score))
                    hands_total.append(hands_a + hands_b if seat_swap else hands_a)

                if writer is not None and scores:
                    s = torch.tensor(scores, device=device, dtype=torch.float32)
                    writer.add_scalar("selfplay/snapshots_filled", len(eval_ids), env_steps)
                    writer.add_scalar(
                        "selfplay/bb_per_hand_mean", float(s.mean().item()), env_steps
                    )
                    writer.add_scalar("selfplay/bb_per_hand_min", float(s.min().item()), env_steps)
                    writer.add_scalar("selfplay/bb_per_hand_max", float(s.max().item()), env_steps)
                    writer.add_scalar("selfplay/eval_opponents", float(len(scores)), env_steps)
                    writer.add_scalar(
                        "selfplay/eval_hands_total", float(sum(hands_total)), env_steps
                    )

                    # Explicit "vs latest snapshot" headline.
                    snap_indices = [
                        i
                        for i, m in enumerate(opponent_metas[:opponent_filled])
                        if m.kind == "snapshot"
                    ]
                    if snap_indices:
                        latest_i = max(
                            snap_indices,
                            key=lambda i: (
                                opponent_metas[i].env_steps or 0,
                                opponent_metas[i].update or 0,
                            ),
                        )

                        meta = opponent_metas[latest_i]
                        snap_path = os.path.join(cfg.checkpoint_dir, meta.id)
                        if snap_path not in snapshot_models:
                            snapshot_models[snap_path] = load_snapshot_model(
                                snap_path, device, cfg, scalar_dim
                            )
                        snap_net = snapshot_models[snap_path]

                        m_latest = eval_vs_policy(
                            env=eval_env,
                            policy_p0=net,
                            policy_p1=snap_net,
                            target_hands=int(cfg.selfplay_eval_hands),
                            max_steps=cfg.eval_max_steps,
                            deterministic=det,
                        )
                        bb0 = float(m_latest["bb_per_hand"])
                        bb1 = 0.0
                        if seat_swap:
                            m_latest2 = eval_vs_policy(
                                env=eval_env,
                                policy_p0=snap_net,
                                policy_p1=net,
                                target_hands=int(cfg.selfplay_eval_hands),
                                max_steps=cfg.eval_max_steps,
                                deterministic=det,
                            )
                            bb1 = -float(m_latest2["bb_per_hand"])
                        bb_latest = 0.5 * (bb0 + bb1) if seat_swap else bb0
                        writer.add_scalar("selfplay/vs_latest_bb_per_hand", bb_latest, env_steps)
                if cfg.timing_sync:
                    _sync(cfg.device)
                t_selfplay_eval_s = time.perf_counter() - t_sp0
            finally:
                if was_training:
                    net.train()

        if writer is not None and do_log:
            writer.add_scalar("time/collect_s", t_collect_s, env_steps)
            writer.add_scalar("time/opt_s", t_opt_s, env_steps)
            writer.add_scalar("time/eval_s", t_eval_s, env_steps)
            writer.add_scalar("time/league_eval_s", t_league_eval_s, env_steps)
            writer.add_scalar("time/selfplay_eval_s", t_selfplay_eval_s, env_steps)
            writer.add_scalar("time/misc_s", t_misc_s, env_steps)
            writer.add_scalar(
                "time/update_total_s",
                t_collect_s + t_opt_s + t_eval_s + t_league_eval_s + t_selfplay_eval_s + t_misc_s,
                env_steps,
            )

        # tqdm progress bar postfix (fast to compute; avoids per-step noise).
        if do_log:
            update_elapsed = time.perf_counter() - update_t0
            update_sps = (cfg.num_envs * cfg.rollout_steps) / max(1e-9, update_elapsed)
            overall_sps_now = float(env_steps) / max(1e-9, time.perf_counter() - t0)
            # `diag_*` are only computed when `do_log` and `_epoch==0`; keep postfix robust.
            diag_total_f = float(diag_total_t.item()) if do_log else 0.0
            if diag_total_f > 0:
                nf = (diag_old_nonfinite_t + diag_new_nonfinite_t) / diag_total_t.clamp_min(1.0)
                nf_rate = float(nf.item())
            else:
                nf_rate = 0.0
            pbar.set_postfix(
                upd=update,
                sps=f"{update_sps / 1000:.0f}k",
                all=f"{overall_sps_now / 1000:.0f}k",
                tcol=f"{t_collect_s:.2f}",
                topt=f"{t_opt_s:.2f}",
                teval=f"{(t_eval_s + t_league_eval_s + t_selfplay_eval_s):.2f}",
                loss=f"{loss_u:.2f}",
                kl=f"{kl_u:.4f}",
                klabs=f"{kl_abs_u:.4f}",
                clip=f"{clip_u:.3f}",
                ent=f"{ent_u:.3f}",
                ev=f"{ev_rollout:.3f}",
                nf=f"{nf_rate:.2e}",
                eval=f"{eval_metrics.get('bb_per_hand', 0.0):.3f}",
            )
        maybe_refresh("idle")

        update += 1

    pbar.close()

    _sync(cfg.device)
    t1 = time.perf_counter()

    elapsed_s = t1 - t0
    overall_sps = env_steps / elapsed_s if elapsed_s > 0 else float("inf")
    denom = max(1, opt_steps)
    denom_t = torch.tensor(float(denom), device=device, dtype=torch.float32)
    mean_loss = float((loss_sum_t / denom_t).item())
    approx_kl = float((kl_sum_t / denom_t).item())
    clipfrac = float((clipfrac_sum_t / denom_t).item())
    value_loss = float((value_loss_sum_t / denom_t).item())
    policy_loss = float((policy_loss_sum_t / denom_t).item())
    entropy = float((entropy_sum_t / denom_t).item())
    invalid_rate = float(invalid_accum.item()) / max(1, env_steps)

    if writer is not None:  # avoid TB flush thread warnings on exit
        writer.flush()
        writer.close()

    return TrainResult(
        config=asdict(cfg),
        env_steps=env_steps,
        opt_steps=opt_steps,
        elapsed_s=elapsed_s,
        overall_sps=overall_sps,
        mean_loss=mean_loss,
        approx_kl=approx_kl,
        clipfrac=clipfrac,
        value_loss=value_loss,
        policy_loss=policy_loss,
        entropy=entropy,
        invalid_rate=invalid_rate,
        eval=eval_metrics,
    )


# =========================================================================== #
# AlphaStar triad training loop (cfg.triad_enabled == True)
# =========================================================================== #


def _run_impl_triad(cfg: TrainConfig) -> TrainResult:
    """Triad training loop: M / ME / LE on partitioned env slices.

    Sibling of ``_run_impl`` (single-net path). Selected by
    ``run(cfg)`` when ``cfg.triad_enabled=True``. Reuses ``_make_policy``,
    ``_compute_mc_returns_p0``, ``_push_transformer_token``, and the
    UnifiedOpponentPool; everything else is per-role via
    ``TriadController``.

    Simplifications vs ``_run_impl``:
      * AMP is honored but no GradScaler (we run bf16 by default; fp16
        with scaler can be added in a follow-up if V100 perf demands it).
      * No torch.compile / cuda-graph paths (compile interacts poorly
        with the per-role mutate_after_snapshot weight swap).
      * Diagnostics are role-prefixed but simplified to means rather
        than the on-device tensor accumulators used in ``_run_impl``.
      * Async eval is stubbed: ME's WR-vs-current-Main is computed
        synchronously every ``cfg.triad_eval_every_updates`` updates
        from the rollout reward stream (no separate eval workers). LE
        WR vs ALL is also seeded from rollout-time reward streams.
    """
    # ---- cfg normalization (mirrors _run_impl) -----------------------------
    cfg_dict = None
    if not os.path.isabs(cfg.checkpoint_dir):
        cfg_dict = asdict(cfg)
        cfg_dict["checkpoint_dir"] = str(_REPO_ROOT / cfg.checkpoint_dir)
    if cfg.resume_from and not os.path.isabs(cfg.resume_from):
        if cfg_dict is None:
            cfg_dict = asdict(cfg)
        cfg_dict["resume_from"] = str(_REPO_ROOT / cfg.resume_from)
    if cfg_dict is not None:
        cfg = TrainConfig(**cfg_dict)

    torch.manual_seed(cfg.seed)
    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device(cfg.device)
    env = WarpPokerEnv(
        num_envs=cfg.num_envs,
        starting_stack=int(cfg.starting_stack),
        small_blind=int(cfg.small_blind),
        big_blind=int(cfg.big_blind),
        device=cfg.device,
    )
    if int(cfg.starting_stack_min) < int(cfg.starting_stack_max):
        per_env_stack = torch.randint(
            int(cfg.starting_stack_min),
            int(cfg.starting_stack_max) + 1,
            (cfg.num_envs,),
            device=device,
            dtype=torch.int32,
        )
        env.reset_with_config(starting_stack=per_env_stack)
    obs = env.reset()
    scalar_dim = int(obs["scalars"].shape[1])

    # ---- TriadController ---------------------------------------------------
    def _factory() -> PolicyNet:
        return _make_policy(cfg, scalar_dim, device)

    triad = TriadController(cfg=cfg, scalar_dim=scalar_dim, device=device, make_net=_factory)
    if cfg.resume_from:
        triad.init_from_resume(cfg.resume_from)

    # ---- AMP ----------------------------------------------------------------
    amp_dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    if cfg.amp_dtype == "auto":
        cap = torch.cuda.get_device_capability() if cfg.device.startswith("cuda") else (0, 0)
        amp_dtype = torch.bfloat16 if cap >= (8, 0) else torch.float16
    else:
        amp_dtype = amp_dtype_map.get(cfg.amp_dtype, torch.bfloat16)
    amp_enabled = cfg.use_amp and cfg.device.startswith("cuda")

    # ---- Opponent pool (shared across roles) -------------------------------
    pool = UnifiedOpponentPool(
        pool_dir=Path(cfg.checkpoint_dir) / "league_pool",
        bot_names=[],  # triad path never trains against scripted bots
    )
    opponent_metas: list[OpponentMeta] = pool.load_index()
    max_opponents = (
        int(cfg.triad_main_pool_cap)
        + int(cfg.triad_me_pool_cap)
        + int(cfg.triad_le_pool_cap)
        + 16  # headroom for incremental growth between prunes
    )
    opponent_scores = torch.zeros(max_opponents, device=device, dtype=torch.float32)
    for i, m in enumerate(opponent_metas[:max_opponents]):
        opponent_scores[i] = float(m.bb_per_hand) if m.bb_per_hand is not None else 0.0
    snapshot_models: dict[str, PolicyNet] = {}

    def _load_snapshot(meta: OpponentMeta) -> PolicyNet:
        if meta.id in snapshot_models:
            return snapshot_models[meta.id]
        snapshot_models[meta.id] = load_snapshot_model(meta.id, device, cfg, scalar_dim)
        return snapshot_models[meta.id]

    # ---- Per-env per-seat hidden state (single global tensors) -------------
    # The TriadController carries per-role h0/c0/h1/c1 sized [1, slice_n, H],
    # but the env loop needs global [1, num_envs, H] tensors so that opponent
    # forwards (which use per-snapshot state) compose naturally. We keep
    # per-role views via env_slice() and copy back into the role state at
    # rollout boundaries.
    hidden = int(triad.role(ROLE_MAIN).net.lstm_hidden)
    state_dtype = torch.float32
    h0 = torch.zeros((1, cfg.num_envs, hidden), device=device, dtype=state_dtype)
    c0 = torch.zeros((1, cfg.num_envs, hidden), device=device, dtype=state_dtype)
    h1 = torch.zeros((1, cfg.num_envs, hidden), device=device, dtype=state_dtype)
    c1 = torch.zeros((1, cfg.num_envs, hidden), device=device, dtype=state_dtype)
    snap_h0 = torch.zeros_like(h0)
    snap_c0 = torch.zeros_like(c0)

    # ---- Per-env opponent assignment tensors (shared global view) ----------
    opp_kind = torch.zeros((cfg.num_envs,), device=device, dtype=torch.int64)
    opp_id = torch.zeros((cfg.num_envs,), device=device, dtype=torch.int64)
    learner_seat = torch.zeros((cfg.num_envs,), device=device, dtype=torch.int64)

    # Per-env "role of the slice that owns this env" -- precomputed once for
    # fast role-aware indexing during reward attribution.
    env_role_id = torch.zeros((cfg.num_envs,), device=device, dtype=torch.int64)
    for i, role in enumerate(ROLES):
        sl = triad.env_slice(role)
        env_role_id[sl] = i

    def _resample_for_role(role: str, reset_mask_slice: torch.Tensor) -> None:
        """Resample opp_kind / opp_id / learner_seat for terminated envs in
        this role's slice. reset_mask_slice is a bool tensor of length
        slice_n (the role's slice).
        """
        sl = triad.env_slice(role)
        idx_local = torch.nonzero(reset_mask_slice, as_tuple=False).squeeze(-1)
        if idx_local.numel() == 0:
            return
        num = int(idx_local.numel())
        idx_global = idx_local + sl.start
        learner_seat[idx_global] = torch.randint(0, 2, (num,), device=device, dtype=torch.int64)
        # Sample opponents only for the reset subset.
        out = triad.sample_opponents(
            role,
            num,
            opponent_metas=opponent_metas,
            opponent_scores=opponent_scores,
        )
        opp_kind[idx_global] = out["opp_kind"]
        opp_id[idx_global] = out["opp_id"]

    # Initial assignment for all envs.
    for role in ROLES:
        sl = triad.env_slice(role)
        all_reset = torch.ones((sl.stop - sl.start,), device=device, dtype=torch.bool)
        _resample_for_role(role, all_reset)

    # ---- Eval env (real async eval to replace wr_proxy) --------------------
    # Created lazily on first eval fire to avoid disturbing PyTorch's CUDA
    # allocator init path (the broken-NVML host crashes when a new Warp env
    # allocation arrives mid-training).
    eval_env: WarpPokerEnv | None = None

    # ---- TensorBoard -------------------------------------------------------
    writer = None
    if _HAS_TB:
        tb_dir = os.environ.get("POKERGPU_TRAIN_TB_DIR", "train/tensorboard")
        writer = SummaryWriter(log_dir=tb_dir)

    # ---- MLflow ------------------------------------------------------------
    # start_run handles its own failure modes (returns a disabled handle on
    # any error). All subsequent calls are no-ops when disabled; failures
    # during training latch the handle into disabled_after_error so we don't
    # spam retries.
    from train import mlflow_logging as mlf

    mlf_run = mlf.start_run(cfg)

    # ---- Loop bookkeeping --------------------------------------------------
    env_steps_per_rollout = int(cfg.num_envs) * int(cfg.rollout_steps)
    max_updates = cfg.num_updates if cfg.total_env_steps <= 0 else 0
    target_env_steps = cfg.total_env_steps if cfg.total_env_steps > 0 else 0
    pbar_total = (
        target_env_steps
        if target_env_steps > 0
        else (max_updates * env_steps_per_rollout if max_updates > 0 else 0)
    )
    env_steps = 0
    update = 0
    pbar = tqdm(
        total=pbar_total if pbar_total > 0 else None,
        initial=0,
        unit="env_step",
        dynamic_ncols=True,
        # Always enabled. Previously disabled when stdout isn't a TTY (i.e.
        # under `tee` or systemd) which made long triad runs look frozen --
        # no per-tick metric for ~10 PPO updates AND no progress bar.
        disable=False,
    )
    log_stride = cfg.log_every_env_steps if cfg.log_every_env_steps > 0 else env_steps_per_rollout
    next_log_env_steps = env_steps + log_stride
    # Per-update "alive" line: short stdout-only print every
    # cfg.log_every_seconds wall-clock seconds. Independent of MLflow / TB
    # logging (which still fires per log_every_env_steps). Lets the user see
    # SPS + update count without waiting ~10 updates for the first real tick.
    print_interval_s = max(0.5, float(cfg.log_every_seconds))
    next_print_time = time.perf_counter() + print_interval_s
    next_save_env_steps = (
        env_steps + cfg.save_every_env_steps if cfg.save_every_env_steps > 0 else 0
    )

    # Cumulative per-role rollout-reward bookkeeping for the synchronous-eval
    # stub: WR vs current main / WR vs all is approximated from the per-role
    # reward stream. T7 will replace this with proper async evals.
    role_reward_sum: dict[str, float] = dict.fromkeys(ROLES, 0.0)
    role_reward_count: dict[str, float] = dict.fromkeys(ROLES, 0.0)

    t0 = time.perf_counter()

    # Wall-clock attribution helper. CUDA is async, so naive time.perf_counter()
    # between kernel-launching python lines under-counts the GPU phase and
    # over-counts the phase where the work finally synchronizes (usually
    # `.item()` calls during PPO update). With timing_sync enabled we force a
    # device sync at each phase boundary so the time delta reflects real GPU
    # work, not when CPU happened to read the result.
    sync_enabled = bool(cfg.timing_sync) and device.type == "cuda"

    def _t_now() -> float:
        if sync_enabled:
            torch.cuda.synchronize(device)
        return time.perf_counter()

    # Most recent per-role PPO metrics. Declared before the loop so the
    # post-loop final-checkpoint / TrainResult code stays bound even if the
    # loop breaks on its first iteration (target already met / max_updates=0).
    per_role_metrics: dict[str, dict[str, float]] = {}

    # =================== main loop ======================================
    while True:
        if max_updates > 0 and update >= max_updates:
            break
        if target_env_steps > 0 and env_steps >= target_env_steps:
            break

        # Phase timing accumulators for this update. Reported in seconds.
        t_phase_rollout = 0.0
        t_phase_returns = 0.0
        t_phase_ppo = 0.0
        t_phase_eval_proxy = 0.0
        t_phase_snapshot = 0.0
        t_phase_save = 0.0
        t_phase_log = 0.0
        t_upd_start = _t_now()

        # ---- Rollout: collect cfg.rollout_steps env steps -----------------
        _t_rollout_start = _t_now()
        with torch.no_grad():
            for t_step in range(cfg.rollout_steps):
                reset = obs["terminated"].to(dtype=torch.bool)
                if reset.any():
                    z = torch.zeros_like(h0)
                    h0 = torch.where(reset.view(1, -1, 1), z, h0)
                    c0 = torch.where(reset.view(1, -1, 1), z, c0)
                    h1 = torch.where(reset.view(1, -1, 1), z, h1)
                    c1 = torch.where(reset.view(1, -1, 1), z, c1)
                    for role in ROLES:
                        sl = triad.env_slice(role)
                        _resample_for_role(role, reset[sl])

                pid = obs["player_id"].to(dtype=torch.int64)
                ep_id = obs["episode_id"].to(dtype=torch.int32)
                h_batch, c_batch = _select_player_state(player_id=pid, h0=h0, c0=c0, h1=h1, c1=c1)

                # Snapshot s_t into each role's buffer (sliced views).
                for role in ROLES:
                    rs = triad.role(role)
                    sl = triad.env_slice(role)
                    rs.buf.cards[t_step].copy_(obs["cards"][sl])
                    rs.buf.scalars[t_step].copy_(obs["scalars"][sl])
                    rs.buf.action_mask[t_step].copy_(obs["action_mask"][sl])
                    rs.buf.player_id[t_step].copy_(pid[sl])
                    rs.buf.episode_id[t_step].copy_(ep_id[sl])
                    rs.buf.reset_mask[t_step].copy_(reset[sl])
                    rs.buf.h[t_step].copy_(h_batch[0, sl, :])
                    rs.buf.c[t_step].copy_(c_batch[0, sl, :])

                # Decide actor per env: learner is the role's net iff
                # selfplay (opp_kind=SELF) or pid == learner_seat. The
                # OPP_LIVE_MAIN case behaves like a "snapshot" but uses
                # the live main net.
                is_selfplay = opp_kind.eq(OPP_SELF)
                is_learner_actor = is_selfplay | pid.eq(learner_seat)

                action_type = torch.full(
                    (cfg.num_envs,), c.ACTION_FOLD, device=device, dtype=torch.int32
                )
                raise_bucket = torch.zeros((cfg.num_envs,), device=device, dtype=torch.int64)
                logprob = torch.zeros((cfg.num_envs,), device=device, dtype=torch.float32)
                value = torch.zeros((cfg.num_envs,), device=device, dtype=torch.float32)

                # ---- Learner-actor forwards (per role) --------------------
                for role in ROLES:
                    rs = triad.role(role)
                    sl = triad.env_slice(role)
                    learner_mask_slice = is_learner_actor[sl]
                    rs.buf.learner_mask[t_step].copy_(learner_mask_slice)
                    if not bool(learner_mask_slice.any().item()):
                        continue
                    idx_local = torch.nonzero(learner_mask_slice, as_tuple=False).squeeze(-1)
                    idx_global = idx_local + sl.start
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                        out_l = rs.net.forward_step(
                            cards=obs["cards"][idx_global],
                            scalars=obs["scalars"][idx_global],
                            action_mask=obs["action_mask"][idx_global],
                            h=h_batch[:, idx_global, :],
                            c=c_batch[:, idx_global, :],
                            terminated=None,
                            deterministic=False,
                        )
                    action_type[idx_global] = out_l.action_type.to(dtype=torch.int32)
                    raise_bucket[idx_global] = out_l.raise_bucket.to(dtype=raise_bucket.dtype)
                    logprob[idx_global] = out_l.logprob.to(dtype=logprob.dtype)
                    value[idx_global] = out_l.value.to(dtype=value.dtype)
                    # Scatter updated state back to the global h/c tensors.
                    pid_l = pid[idx_global].to(dtype=torch.bool)
                    if (~pid_l).any():
                        idx0 = idx_global[~pid_l]
                        h0[:, idx0, :] = out_l.h[:, ~pid_l, :]
                        c0[:, idx0, :] = out_l.c[:, ~pid_l, :]
                    if pid_l.any():
                        idx1 = idx_global[pid_l]
                        h1[:, idx1, :] = out_l.h[:, pid_l, :]
                        c1[:, idx1, :] = out_l.c[:, pid_l, :]

                # ---- Opponent-actor forwards -----------------------------
                is_opp_actor = (~is_learner_actor) & (~is_selfplay)
                # OPP_LIVE_MAIN: opponent is the live Main net (ME's case).
                live_main_mask = is_opp_actor & opp_kind.eq(OPP_LIVE_MAIN)
                if live_main_mask.any():
                    idx_lm = torch.nonzero(live_main_mask, as_tuple=False).squeeze(-1)
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                        out_lm = triad.role(ROLE_MAIN).net.forward_step(
                            cards=obs["cards"][idx_lm],
                            scalars=obs["scalars"][idx_lm],
                            action_mask=obs["action_mask"][idx_lm],
                            h=h_batch[:, idx_lm, :],
                            c=c_batch[:, idx_lm, :],
                            terminated=None,
                            deterministic=True,
                        )
                    action_type[idx_lm] = out_lm.action_type.to(dtype=torch.int32)
                    raise_bucket[idx_lm] = out_lm.raise_bucket.to(dtype=raise_bucket.dtype)
                    # Scatter opponent state into the per-seat tensors so its
                    # NEXT decision sees a coherent history.
                    pid_lm = pid[idx_lm].to(dtype=torch.bool)
                    if (~pid_lm).any():
                        idx0 = idx_lm[~pid_lm]
                        h0[:, idx0, :] = out_lm.h[:, ~pid_lm, :]
                        c0[:, idx0, :] = out_lm.c[:, ~pid_lm, :]
                    if pid_lm.any():
                        idx1 = idx_lm[pid_lm]
                        h1[:, idx1, :] = out_lm.h[:, pid_lm, :]
                        c1[:, idx1, :] = out_lm.c[:, pid_lm, :]

                # OPP_SNAPSHOT: load and forward each unique snapshot once.
                snap_mask = is_opp_actor & opp_kind.eq(OPP_SNAPSHOT)
                if snap_mask.any():
                    for pool_idx in torch.unique(opp_id[snap_mask]).tolist():
                        pool_idx = int(pool_idx)
                        if pool_idx >= len(opponent_metas):
                            continue
                        meta = opponent_metas[pool_idx]
                        if meta.kind != "snapshot":
                            continue
                        m = snap_mask & opp_id.eq(pool_idx)
                        if not m.any():
                            continue
                        idx_s = torch.nonzero(m, as_tuple=False).squeeze(-1)
                        snap_net = _load_snapshot(meta)
                        with torch.autocast(
                            device_type="cuda", dtype=amp_dtype, enabled=amp_enabled
                        ):
                            out_s = snap_net.forward_step(
                                cards=obs["cards"][idx_s],
                                scalars=obs["scalars"][idx_s],
                                action_mask=obs["action_mask"][idx_s],
                                h=snap_h0[:, idx_s, :],
                                c=snap_c0[:, idx_s, :],
                                terminated=None,
                                deterministic=True,
                            )
                        action_type[idx_s] = out_s.action_type.to(dtype=torch.int32)
                        raise_bucket[idx_s] = out_s.raise_bucket.to(dtype=raise_bucket.dtype)

                # ---- env.step -----------------------------------------
                amounts = raise_bucket_to_amount(
                    raise_bucket=raise_bucket,
                    pot_total=obs["pot_total"],
                    min_raise=obs["min_raise"],
                    max_raise=obs["max_raise"],
                )
                amounts = torch.where(
                    action_type.eq(c.ACTION_RAISE),
                    amounts,
                    torch.zeros_like(amounts),
                )
                next_obs = env.step(action_type, amounts)

                if cfg.policy_backbone == "transformer":
                    stage_at_action = _stage_from_scalars(obs["scalars"])
                    h0, h1 = _push_transformer_token(
                        h0,
                        h1,
                        player_id=pid,
                        action_type=action_type,
                        raise_bucket=raise_bucket,
                        stage=stage_at_action,
                    )

                bb = next_obs["big_blind"].to(dtype=torch.float32).clamp_min(1.0)
                reward_p0_full = next_obs["rewards"].to(dtype=torch.float32) / (
                    bb * float(c.DEFAULT_STACK_BB)
                )
                reward_p0_full = torch.where(
                    next_obs["terminated"].to(dtype=torch.bool),
                    reward_p0_full,
                    torch.zeros_like(reward_p0_full),
                )

                # Distribute (a_t, logp_t, v_t, r_t, done_t) into each role's buffer.
                for role in ROLES:
                    rs = triad.role(role)
                    sl = triad.env_slice(role)
                    rs.buf.action_type[t_step].copy_(action_type[sl].to(dtype=torch.int64))
                    rs.buf.raise_bucket[t_step].copy_(raise_bucket[sl])
                    rs.buf.logprob[t_step].copy_(logprob[sl])
                    rs.buf.value[t_step].copy_(value[sl])
                    rs.buf.reward[t_step].zero_()
                    rs.buf.reward_p0[t_step].copy_(reward_p0_full[sl])
                    rs.buf.done[t_step].copy_(next_obs["terminated"][sl])

                obs = next_obs

        env_steps += env_steps_per_rollout
        pbar.update(env_steps_per_rollout)
        t_phase_rollout = _t_now() - _t_rollout_start

        # ---- PPO update: per-role GAE returns + advantages ---------------
        # Fix for the EV-collapse bug: triad previously used MC returns
        # (terminal-only, high-variance). Now uses GAE on per-step values.
        # The rollout populates rs.buf.value only for learner-actor steps;
        # we backfill values for opponent-actor steps via a post-hoc no-grad
        # forward through the role's net so GAE has a valid bootstrap at
        # every state. last_value bootstraps the truncated rollout window.
        _t_returns_start = _t_now()

        # Backfill values at opponent-actor steps (currently 0 in buf.value).
        # Chunked to keep peak activation memory bounded: at full num_envs
        # the full opp batch (T * frac_opp * slice_n) can OOM the transformer
        # encoder. Chunk size matches the rollout's per-step learner batch
        # (~slice_n samples) so activation memory stays at rollout-scale.
        backfill_chunk = max(1024, int(cfg.num_envs))
        with torch.no_grad():
            for role in ROLES:
                rs = triad.role(role)
                sl = triad.env_slice(role)
                opp_mask = ~rs.buf.learner_mask  # [T, slice_n] bool
                if not bool(opp_mask.any().item()):
                    continue
                idx_t, idx_n = torch.nonzero(opp_mask, as_tuple=True)
                n_opp = int(idx_t.numel())
                for start in range(0, n_opp, backfill_chunk):
                    end = min(start + backfill_chunk, n_opp)
                    s_t = idx_t[start:end]
                    s_n = idx_n[start:end]
                    h_gath = rs.buf.h[s_t, s_n].unsqueeze(0)
                    c_gath = rs.buf.c[s_t, s_n].unsqueeze(0)
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                        out_v = rs.net.forward_step(
                            cards=rs.buf.cards[s_t, s_n],
                            scalars=rs.buf.scalars[s_t, s_n],
                            action_mask=rs.buf.action_mask[s_t, s_n],
                            h=h_gath,
                            c=c_gath,
                            terminated=None,
                            deterministic=True,
                        )
                    rs.buf.value[s_t, s_n] = out_v.value.to(dtype=rs.buf.value.dtype)

            # Compute last_value (bootstrap for the state AFTER the rollout
            # window) per role. `obs` here is the next-step observation
            # post-loop; the role's hidden state to use is whatever h0/c0/h1/c1
            # currently hold for the appropriate seat. Use seat0's state as
            # the bootstrap reference -- correct for episodes that end inside
            # the rollout (dones[T-1] == True), and an approximation otherwise
            # (only matters for the unfinished tail).
            last_value_by_role: dict[str, torch.Tensor] = {}
            last_player_id_by_role: dict[str, torch.Tensor] = {}
            pid_now = obs["player_id"].to(dtype=torch.int64)
            for role in ROLES:
                sl = triad.env_slice(role)
                pid_sl = pid_now[sl]
                h_b = torch.where(pid_sl.view(1, -1, 1).eq(0), h0[:, sl, :], h1[:, sl, :])
                c_b = torch.where(pid_sl.view(1, -1, 1).eq(0), c0[:, sl, :], c1[:, sl, :])
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                    out_last = triad.role(role).net.forward_step(
                        cards=obs["cards"][sl],
                        scalars=obs["scalars"][sl],
                        action_mask=obs["action_mask"][sl],
                        h=h_b,
                        c=c_b,
                        terminated=None,
                        deterministic=True,
                    )
                last_value_by_role[role] = out_last.value.detach().to(dtype=torch.float32)
                last_player_id_by_role[role] = pid_sl.clone()

        returns_by_role: dict[str, torch.Tensor] = {}
        adv_by_role: dict[str, torch.Tensor] = {}
        mask_by_role: dict[str, torch.Tensor | None] = {}
        per_role_metrics: dict[str, dict[str, float]] = {}
        for role in ROLES:
            rs = triad.role(role)
            # Per-step actor-perspective reward: flip sign when p1 was actor.
            reward_actor = torch.where(rs.buf.player_id.eq(0), rs.buf.reward_p0, -rs.buf.reward_p0)
            advantages, returns = _compute_gae(
                rewards=reward_actor,
                dones=rs.buf.done.to(dtype=torch.bool),
                values=rs.buf.value,
                last_value=last_value_by_role[role],
                player_id=rs.buf.player_id.to(dtype=torch.int64),
                last_player_id=last_player_id_by_role[role],
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
            )
            train_mask = rs.buf.learner_mask
            if not bool(train_mask.any().item()):
                continue
            # Normalize advantages under the train_mask (per-role, per-update).
            w = train_mask.to(dtype=torch.float32)
            denom = w.sum().clamp_min(1.0)
            adv_mean = (advantages * w).sum() / denom
            adv_var = ((advantages - adv_mean).pow(2) * w).sum() / denom
            adv_std = adv_var.sqrt().clamp_min(1e-6)
            adv_norm = (advantages - adv_mean) / adv_std

            returns_by_role[role] = returns
            adv_by_role[role] = adv_norm
            mask_by_role[role] = train_mask

            # Accumulate role reward stats (for the synchronous WR-stub).
            terminal_rewards = rs.buf.reward_p0[rs.buf.done.to(dtype=torch.bool)]
            if terminal_rewards.numel() > 0:
                role_reward_sum[role] += float(terminal_rewards.sum().item())
                role_reward_count[role] += float(terminal_rewards.numel())

        t_phase_returns = _t_now() - _t_returns_start

        _t_ppo_start = _t_now()
        if returns_by_role:
            per_role_metrics = triad.ppo_update(
                returns_by_role=returns_by_role,
                advantages_by_role=adv_by_role,
                train_mask_by_role=mask_by_role,
            )
        t_phase_ppo = _t_now() - _t_ppo_start

        # ---- Real eval: ME/LE vs current Main (every triad_eval_every_updates)
        # Replaces the prior sigmoid-of-reward proxy with actual hand-by-hand
        # play. Promotes ME/LE snapshots only when their REAL win rate vs the
        # main agent crosses triad_me_promote_wr / triad_le_promote_wr.
        _t_eval_start = _t_now()
        triad_eval_every = max(1, int(cfg.triad_eval_every_updates))
        if (update + 1) % triad_eval_every == 0:
            if eval_env is None:
                eval_env = WarpPokerEnv(
                    num_envs=min(1024, int(cfg.num_envs)),
                    starting_stack=int(cfg.starting_stack),
                    small_blind=int(cfg.small_blind),
                    big_blind=int(cfg.big_blind),
                    device=cfg.device,
                )
            main_net = triad.role(ROLE_MAIN).net
            main_net.eval()
            for role in (ROLE_ME, ROLE_LE):
                rs = triad.role(role)
                rs.net.eval()
                try:
                    # Average across both seat assignments: role as P0 then as P1.
                    m_a = eval_vs_policy(
                        env=eval_env,
                        policy_p0=rs.net,
                        policy_p1=main_net,
                        target_hands=int(cfg.triad_eval_hands),
                        max_steps=int(cfg.eval_max_steps),
                        deterministic=False,
                    )
                    bb_a = float(m_a.get("bb_per_hand", 0.0))
                    m_b = eval_vs_policy(
                        env=eval_env,
                        policy_p0=main_net,
                        policy_p1=rs.net,
                        target_hands=int(cfg.triad_eval_hands),
                        max_steps=int(cfg.eval_max_steps),
                        deterministic=False,
                    )
                    bb_b = -float(m_b.get("bb_per_hand", 0.0))
                    bb_avg = 0.5 * (bb_a + bb_b)
                    # Map bb/hand -> [0, 1] WR via same sigmoid as the old
                    # proxy so triad_me_promote_wr=0.70 (etc.) keeps its
                    # configured semantics. sigmoid(5x) crosses 0.7 at
                    # bb_per_hand ~= 0.17 = +17 bb/100.
                    wr = float(torch.sigmoid(torch.tensor(bb_avg * 5.0)).item())
                    rs.eval_wr_vs_targets["main_current"] = wr
                    # Also emit raw bb/hand for the user-facing metric.
                    rs.eval_wr_vs_targets["main_current_bb_per_hand"] = bb_avg
                finally:
                    rs.net.train()
            main_net.train()

        t_phase_eval_proxy = _t_now() - _t_eval_start

        # ---- Snapshot triggers -------------------------------------------
        _t_snap_start = _t_now()
        decisions = triad.maybe_snapshot(update, env_steps)
        for role, dec in decisions.items():
            if not dec.should_snapshot:
                continue
            rs = triad.role(role)
            meta = pool.add_snapshot(
                model=rs.net,
                env_steps=env_steps,
                update=update,
                agent_role=dec.agent_role_tag,
            )
            # Track in-memory pool view.
            opponent_metas.append(meta)
            if len(opponent_metas) > opponent_scores.numel():
                # Grow scores tensor as the pool grows beyond initial headroom.
                new_size = opponent_scores.numel() * 2
                new_scores = torch.zeros(new_size, device=device, dtype=torch.float32)
                new_scores[: opponent_scores.numel()] = opponent_scores
                opponent_scores = new_scores
            opponent_scores[len(opponent_metas) - 1] = 0.0
            rs.last_snapshot_update = update
            applied = triad.mutate_after_snapshot(role)
            if writer is not None:
                writer.add_scalar(f"{role}/snapshot_count", float(rs.snapshot_count), env_steps)
                writer.add_scalar(f"{role}/snapshot_applied_{applied}", 1.0, env_steps)
            mlf.log_snapshot_event(
                mlf_run,
                role=role,
                snapshot_count=rs.snapshot_count,
                env_steps=env_steps,
                update=update,
                mutate_applied=applied,
            )
            # Snapshot upload only in "all" mode -- the snapshot pool can
            # grow to dozens of files per role over a long run.
            if cfg.mlflow_upload_checkpoints == "all":
                mlf.log_checkpoint(
                    mlf_run,
                    meta.id,
                    artifact_path=f"checkpoints/snapshots/{role}",
                )

        # ---- ME forced periodic reset (AlphaStar §3 stalling fix) --------
        # Counters the [[me-exploiter-stalling]] failure mode: once Main
        # catches up to ME, ME's WR collapses below triad_me_promote_wr
        # and the WR-gated snapshot/mutate path stops firing. Without a
        # forced reset, ME stays frozen indefinitely. No-op when
        # cfg.triad_me_force_reset_every <= 0.
        forced = triad.maybe_force_reset_me(update)
        if forced is not None:
            rs_me = triad.role(ROLE_ME)
            if writer is not None:
                writer.add_scalar("me/force_reset_count", 1.0, env_steps)
                writer.add_scalar(
                    "me/force_reset_updates_since", float(forced["updates_since"]), env_steps
                )
            mlf.log_snapshot_event(
                mlf_run,
                role=ROLE_ME,
                snapshot_count=rs_me.snapshot_count,
                env_steps=env_steps,
                update=update,
                mutate_applied=str(forced["kind"]),
            )

        t_phase_snapshot = _t_now() - _t_snap_start

        # ---- Periodic checkpoint -----------------------------------------
        _t_save_start = _t_now()
        if cfg.save_every_env_steps > 0 and env_steps >= next_save_env_steps:
            ckpt_path = Path(cfg.checkpoint_dir) / f"triad_main_env{env_steps}_upd{update}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            # opt_steps for the triad path = sum of per-role mb_counts for
            # this last update (coarse but the field is only used for
            # logging / display on resume so exact accuracy isn't critical).
            opt_steps_so_far = int(sum(m.get("mb_count", 0) for m in per_role_metrics.values()))
            save_checkpoint(
                path=str(ckpt_path),
                model=triad.role(ROLE_MAIN).net,
                optimizer=triad.role(ROLE_MAIN).opt,
                state=CheckpointState(
                    env_steps=env_steps,
                    opt_steps=opt_steps_so_far,
                    update=update,
                ),
            )
            if cfg.mlflow_upload_checkpoints in ("periodic", "all"):
                mlf.log_checkpoint(
                    mlf_run,
                    ckpt_path,
                    artifact_path="checkpoints/main/periodic",
                )
            next_save_env_steps += cfg.save_every_env_steps
        t_phase_save = _t_now() - _t_save_start

        # ---- Logging ------------------------------------------------------
        _t_log_start = _t_now()
        do_log = env_steps >= next_log_env_steps
        if do_log:
            overall_sps_now = float(env_steps) / max(1e-9, time.perf_counter() - t0)

            # Assemble the full batched MLflow metric dict (~92 entries).
            # TB writes happen alongside; the two sinks are independent.
            mlf_metrics: dict[str, float] = {
                "train/overall_sps": overall_sps_now,
                "train/update": float(update),
            }
            mlf_metrics.update(mlf.system_stats(cfg.device))

            if writer is not None:
                writer.add_scalar("train/overall_sps", overall_sps_now, env_steps)
                writer.add_scalar("train/update", float(update), env_steps)

            # Per-role PPO metrics.
            for role, m in per_role_metrics.items():
                for k, v in m.items():
                    mlf_metrics[f"{role}/{k}"] = float(v)
                    if writer is not None:
                        writer.add_scalar(f"{role}/{k}", float(v), env_steps)

            # Per-role snapshot bookkeeping + eval WR + distributions.
            for role in ROLES:
                rs = triad.role(role)
                sl = triad.env_slice(role)
                mlf_metrics[f"{role}/snapshot_count"] = float(rs.snapshot_count)
                mlf_metrics[f"{role}/updates_since_last_snapshot"] = float(
                    update - rs.last_snapshot_update if rs.last_snapshot_update >= 0 else update
                )
                if writer is not None:
                    writer.add_scalar(f"{role}/snapshot_count", float(rs.snapshot_count), env_steps)

                for target_id, wr in rs.eval_wr_vs_targets.items():
                    mlf_metrics[f"{role}/eval_wr/{target_id}"] = float(wr)
                    if writer is not None:
                        writer.add_scalar(f"{role}/eval_wr/{target_id}", float(wr), env_steps)

                # Opponent-mix breakdown (from the *current* live assignment;
                # cheap to compute and meaningful as a recent-state snapshot).
                opp_dist = mlf.opp_kind_distribution(opp_kind[sl])
                for k, v in opp_dist.items():
                    mlf_metrics[f"{role}/{k}"] = v
                    if writer is not None:
                        writer.add_scalar(f"{role}/{k}", v, env_steps)

                # Action + raise-bucket distributions over the role's buffer.
                act_dist = mlf.action_distribution(rs.buf.action_type, rs.buf.learner_mask)
                rb_dist = mlf.raise_bucket_distribution(
                    rs.buf.raise_bucket, rs.buf.action_type, rs.buf.learner_mask
                )
                for k, v in {**act_dist, **rb_dist}.items():
                    mlf_metrics[f"{role}/{k}"] = v
                    if writer is not None:
                        writer.add_scalar(f"{role}/{k}", v, env_steps)

                # Per-role terminal-reward stats (already accumulated above).
                n = role_reward_count[role]
                if n > 0:
                    mean_r = role_reward_sum[role] / n
                    mlf_metrics[f"{role}/terminal_reward_mean"] = mean_r
                    mlf_metrics[f"{role}/terminal_hands_count"] = float(n)
                    if writer is not None:
                        writer.add_scalar(f"{role}/terminal_reward_mean", mean_r, env_steps)

            # League pool size breakdown.
            pool_sizes = {"total": len(opponent_metas)}
            for tag in ("main_historical", "main_exploiter", "league_exploiter"):
                pool_sizes[tag] = sum(1 for m in opponent_metas if m.agent_role == tag)
            for k, v in pool_sizes.items():
                mlf_metrics[f"league/pool_size_{k}"] = float(v)
                if writer is not None:
                    writer.add_scalar(f"league/pool_size_{k}", float(v), env_steps)

            # Triad-specific diagnostics.
            me_wr = triad.role(ROLE_ME).eval_wr_vs_targets.get("main_current", 0.0)
            mlf_metrics["triad/me_curriculum_active"] = (
                1.0 if float(me_wr) < float(cfg.triad_me_curriculum_threshold) else 0.0
            )

            # Phase wall-clock attribution (the missing time/*_s block from
            # the non-triad path). Per-update breakdown so a SPS regression
            # can be attributed without re-instrumenting + re-running.
            t_update_total = _t_now() - t_upd_start
            t_phase_log = _t_now() - _t_log_start  # nearly full block
            t_phase_misc = max(
                0.0,
                t_update_total
                - (
                    t_phase_rollout
                    + t_phase_returns
                    + t_phase_ppo
                    + t_phase_eval_proxy
                    + t_phase_snapshot
                    + t_phase_save
                    + t_phase_log
                ),
            )
            mlf_metrics["time/rollout_s"] = t_phase_rollout
            mlf_metrics["time/returns_s"] = t_phase_returns
            mlf_metrics["time/ppo_s"] = t_phase_ppo
            mlf_metrics["time/eval_proxy_s"] = t_phase_eval_proxy
            mlf_metrics["time/snapshot_s"] = t_phase_snapshot
            mlf_metrics["time/save_s"] = t_phase_save
            mlf_metrics["time/log_s"] = t_phase_log
            mlf_metrics["time/misc_s"] = t_phase_misc
            mlf_metrics["time/update_total_s"] = t_update_total
            mlf_metrics["throughput/sps_rollout"] = (
                env_steps_per_rollout / t_phase_rollout if t_phase_rollout > 0 else 0.0
            )
            mlf_metrics["throughput/sps_update"] = (
                env_steps_per_rollout / t_update_total if t_update_total > 0 else 0.0
            )
            if writer is not None:
                for k in (
                    "rollout_s",
                    "returns_s",
                    "ppo_s",
                    "eval_proxy_s",
                    "snapshot_s",
                    "save_s",
                    "log_s",
                    "misc_s",
                    "update_total_s",
                ):
                    writer.add_scalar(f"time/{k}", mlf_metrics[f"time/{k}"], env_steps)

            mlf.log_metrics(mlf_run, mlf_metrics, step=env_steps)
            next_log_env_steps += log_stride
        else:
            # Still close the log-phase timer even when no logging fires, so
            # the "alive" line below has a valid t_update_total.
            t_phase_log = _t_now() - _t_log_start

        update += 1

        # Per-update "alive" line (stdout only; MLflow/TB are gated above).
        now_t = _t_now()
        if now_t >= next_print_time:
            sps_now = float(env_steps) / max(1e-9, now_t - t0)
            roles_summary = " ".join(
                f"{r.upper()}={per_role_metrics.get(r, {}).get('mb_count', 0):.0f}mb" for r in ROLES
            )
            # Compact phase breakdown so the slowest phase is visible per update.
            t_update_total = (
                t_phase_rollout
                + t_phase_returns
                + t_phase_ppo
                + t_phase_eval_proxy
                + t_phase_snapshot
                + t_phase_save
                + t_phase_log
            )
            phase_break = (
                f"roll={t_phase_rollout:.2f}s "
                f"ret={t_phase_returns:.2f}s "
                f"ppo={t_phase_ppo:.2f}s "
                f"eval={t_phase_eval_proxy:.2f}s "
                f"snap={t_phase_snapshot:.2f}s "
                f"save={t_phase_save:.2f}s "
                f"log={t_phase_log:.2f}s "
                f"tot={t_update_total:.2f}s"
            )
            print(
                f"[upd {update:>5}] env={env_steps:>10}  sps={sps_now:>7.0f}  "
                f"{roles_summary}  |  {phase_break}",
                flush=True,
            )
            next_print_time = now_t + print_interval_s

    pbar.close()
    if writer is not None:
        writer.flush()
        writer.close()

    # Final main-checkpoint upload (any non-"none" mode). Always writes a
    # final.pt to local disk first so the upload reflects the EXACT weights
    # at clean exit, independent of save_every_env_steps cadence.
    if cfg.mlflow_upload_checkpoints in ("final", "periodic", "all"):
        final_ckpt = Path(cfg.checkpoint_dir) / f"triad_main_final_env{env_steps}_upd{update}.pt"
        final_ckpt.parent.mkdir(parents=True, exist_ok=True)
        final_opt_steps = int(sum(m.get("mb_count", 0) for m in per_role_metrics.values()))
        save_checkpoint(
            path=str(final_ckpt),
            model=triad.role(ROLE_MAIN).net,
            optimizer=triad.role(ROLE_MAIN).opt,
            state=CheckpointState(
                env_steps=env_steps,
                opt_steps=final_opt_steps,
                update=update,
            ),
        )
        mlf.log_checkpoint(
            mlf_run,
            final_ckpt,
            artifact_path="checkpoints/main/final",
        )

    # MLflow run closes cleanly here -- if we reached this point the loop
    # exited normally (target_env_steps / max_updates). Crashes / KB
    # interrupts bypass this and the run shows up as a "killed" status in
    # the UI (MLflow auto-tags incomplete runs when the process dies).
    mlf.end_run(mlf_run, status="completed")

    elapsed_s = time.perf_counter() - t0
    overall_sps = float(env_steps) / max(1e-9, elapsed_s)
    # Surface the most recent main-role metrics as the headline numbers.
    last_main = per_role_metrics.get(ROLE_MAIN, {})
    return TrainResult(
        config=asdict(cfg),
        env_steps=env_steps,
        opt_steps=int(sum(m.get("mb_count", 0) for m in per_role_metrics.values())),
        elapsed_s=elapsed_s,
        overall_sps=overall_sps,
        mean_loss=float(last_main.get("loss", 0.0)),
        approx_kl=float(last_main.get("approx_kl", 0.0)),
        clipfrac=float(last_main.get("clipfrac", 0.0)),
        value_loss=float(last_main.get("value_loss", 0.0)),
        policy_loss=float(last_main.get("policy_loss", 0.0)),
        entropy=float(last_main.get("entropy", 0.0)),
        invalid_rate=0.0,
        eval={},
    )


def _autotune_num_envs(cfg: TrainConfig) -> TrainConfig:
    """Pick the largest safe ``num_envs`` for the current device + config.

    Uses two-point memory calibration (``train.autotune.autotune_num_envs``)
    that:
      - allocates the configured policy backbone (LSTM or transformer)
      - allocates ``n_concurrent_nets`` copies (3 for triad mode, 1 otherwise)
        to match Main+ME+LE concurrent residency
      - runs a multi-epoch forward/backward to approximate PPO update peak
      - measures ``cuda.max_memory_allocated`` at two small probe sizes,
        fits ``M(N) = fixed + per_env * N``, then solves for the largest
        N that leaves a safety margin under total VRAM.

    Replaces the prior OK/fail binary search which under-counted the real
    training peak by ~4x on triad+transformer and forced manual fallback.
    """
    import dataclasses

    from train.autotune import autotune_num_envs

    n_nets = 3 if cfg.triad_enabled else 1
    print(
        f"[autotune] calibrating on {cfg.device} "
        f"(backbone={cfg.policy_backbone}, n_concurrent_nets={n_nets}, "
        f"ceiling={cfg.num_envs}) ..."
    )
    best = autotune_num_envs(
        device=cfg.device,
        rollout_steps=cfg.rollout_steps,
        include_eval_env=cfg.eval_every > 0,
        n_concurrent_nets=n_nets,
        policy_backbone=cfg.policy_backbone,
        card_embed_dim=cfg.card_embed_dim,
        mlp_dim=cfg.mlp_dim,
        torso_layers=cfg.torso_layers,
        lstm_hidden=cfg.lstm_hidden,
        head_layers=cfg.head_layers,
        head_dim=cfg.head_dim,
        transformer_d_model=cfg.transformer_d_model,
        transformer_n_heads=cfg.transformer_n_heads,
        transformer_n_layers=cfg.transformer_n_layers,
        transformer_ffn_dim=cfg.transformer_ffn_dim,
        num_epochs=cfg.num_epochs,
        ceiling=cfg.num_envs,
    )
    new_minibatch = max(256, (min(cfg.minibatch_size, best) // 256) * 256)
    print(f"[autotune] -> num_envs={best}  minibatch_size={new_minibatch}")
    return dataclasses.replace(cfg, num_envs=best, minibatch_size=new_minibatch, autotune=False)


def main() -> int:
    if not _HAS_HYDRA:  # pragma: no cover
        raise ModuleNotFoundError("hydra-core is not installed")

    # --- Process-wide perf knobs ---------------------------------------------
    # cudnn.benchmark: rollout + PPO use fixed shapes (num_envs x rollout_steps
    # x ...) — let cuDNN pick the fastest convolution algorithm once, cache it.
    # Cheap when shapes are stable (our case); only a problem with variable
    # shapes which we don't have.
    torch.backends.cudnn.benchmark = True
    # TF32 for fp32 matmuls (value head + any residual fp32 path under AMP).
    # On Ampere+ (sm_80+) this is a ~2x speedup for matmul with no measurable
    # accuracy hit on RL workloads. Sm_70 (V100) ignores it gracefully.
    torch.set_float32_matmul_precision("high")

    ConfigStore.instance().store(name="train_config", node=TrainConfig)
    conf_dir = str((_REPO_ROOT / "conf").resolve())

    @hydra.main(version_base=None, config_path=conf_dir, config_name="train")
    def _hydra_main(cfg_dict) -> int:
        cfg_obj = TrainConfig(**OmegaConf.to_container(cfg_dict, resolve=True))  # type: ignore[arg-type]
        if cfg_obj.autotune:
            cfg_obj = _autotune_num_envs(cfg_obj)
        result = run(cfg_obj)

        payload = {
            "meta": {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "host": socket.gethostname(),
                "torch": getattr(torch, "__version__", "unknown"),
            },
            "result": asdict(result),
            "hydra": OmegaConf.to_container(cfg_dict, resolve=True),  # type: ignore[arg-type]
        }

        # Write into the Hydra run dir (CWD), and also update the repo-level "latest".
        Path("results.json").write_text(json.dumps(payload, indent=2) + "\n")
        (_REPO_ROOT / "train/latest_train.json").write_text(json.dumps(payload, indent=2) + "\n")

        print("Wrote: results.json")
        print(f"overall_SPS={result.overall_sps:.2f} mean_loss={result.mean_loss:.6f}")
        if result.eval:
            print(f"eval={result.eval}")
        return 0

    return _hydra_main()  # type: ignore[func-returns-value]


if __name__ == "__main__":
    raise SystemExit(main())
