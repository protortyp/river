"""Training configuration, result types, and shared setup helpers."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class TrainConfig:
    device: str
    num_envs: int
    rollout_steps: int
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
    # League / opponent sampling
    league_enabled: bool
    league_eta_selfplay: float
    league_bot_initial_score: float
    league_eval_bots: bool
    league_eval_bots_every: int
    league_snapshot_pool: int
    league_snapshot_every_updates: int
    league_eval_every_updates: int
    league_eval_hands: int
    league_eval_subset: int
    league_eval_deterministic: bool
    league_eval_seat_swap: bool
    league_pfsp_temperature: float
    league_pfsp_epsilon: float
    league_rollout_snapshot_k: int
    # Self-play evaluation vs snapshot pool
    selfplay_eval_enabled: bool
    selfplay_eval_every_updates: int
    selfplay_eval_hands: int
    selfplay_eval_subset: int
    selfplay_eval_deterministic: bool
    selfplay_eval_seat_swap: bool
    # Runtime / performance knobs
    timing_sync: bool
    compile_policy: bool
    use_amp: bool
    amp_dtype: str
    use_cuda_graph: bool
    autotune: bool = False
    policy_backbone: str = "lstm"
    transformer_d_model: int = 256
    transformer_n_heads: int = 4
    transformer_n_layers: int = 4
    transformer_ffn_dim: int = 1024
    league_pfsp_mode: str = "hard"
    league_pfsp_q: float = 2.0
    league_train_on_bots: bool = False
    starting_stack: int = 1000
    starting_stack_min: int = 1000
    starting_stack_max: int = 1000
    small_blind: int = 5
    big_blind: int = 10

    # AlphaStar triad
    triad_enabled: bool = False
    triad_main_frac: float = 0.6
    triad_me_frac: float = 0.2
    triad_le_frac: float = 0.2
    triad_main_snapshot_every: int = 25
    triad_me_promote_wr: float = 0.70
    triad_le_promote_wr: float = 0.70
    triad_le_reset_prob: float = 0.50
    triad_me_curriculum_threshold: float = 0.30
    triad_me_force_reset_every: int = 0
    triad_eval_every_updates: int = 50
    triad_eval_hands: int = 1000
    triad_main_mix_self: float = 0.35
    triad_main_mix_pfsp: float = 0.50
    triad_main_mix_forgotten: float = 0.15
    triad_main_pool_cap: int = 16
    triad_le_pool_cap: int = 8
    triad_me_pool_cap: int = 0
    triad_lr_main: float | None = None
    triad_lr_me: float | None = None
    triad_lr_le: float | None = None
    triad_entropy_me: float | None = None
    triad_entropy_le: float | None = None

    # R-NaD (Regularized Nash Dynamics) & CTDE omniscient critic
    rnad_enabled: bool = False
    rnad_eta: float = 0.015
    rnad_reg_update_every: int = 40
    rnad_kl_clip: float = 10.0
    use_omniscient_critic: bool = True
    aux_opp_card_coef: float = 0.0
    lr_end: float | None = None
    entropy_coef_end: float | None = None

    # MLflow tracking (opt-in)
    mlflow_enabled: bool = False
    mlflow_experiment: str | None = None
    mlflow_run_name: str | None = None
    mlflow_tags: list[str] | None = None
    mlflow_notes: str | None = None
    mlflow_extra_tags: dict[str, str] | None = None
    mlflow_upload_checkpoints: str = "none"

    def __post_init__(self) -> None:
        valid_upload_modes = {"none", "final", "periodic", "all"}
        if self.mlflow_upload_checkpoints not in valid_upload_modes:
            raise ValueError(
                f"mlflow_upload_checkpoints must be one of {sorted(valid_upload_modes)}, "
                f"got {self.mlflow_upload_checkpoints!r}"
            )

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
            # 64 envs is the floor below which per-role PPO minibatches are too noisy.
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


def normalize_cfg_paths(cfg: TrainConfig, repo_root: Path) -> TrainConfig:
    """Resolve repo-relative checkpoint/resume paths before Hydra changes CWD."""
    cfg_dict = None
    if not os.path.isabs(cfg.checkpoint_dir):
        cfg_dict = asdict(cfg)
        cfg_dict["checkpoint_dir"] = str(repo_root / cfg.checkpoint_dir)
    if cfg.resume_from and not os.path.isabs(cfg.resume_from):
        if cfg_dict is None:
            cfg_dict = asdict(cfg)
        cfg_dict["resume_from"] = str(repo_root / cfg.resume_from)
    return TrainConfig(**cfg_dict) if cfg_dict is not None else cfg


def resolve_amp(cfg: TrainConfig) -> tuple[bool, torch.dtype]:
    """Return (amp_enabled, amp_dtype) using native bf16 on Ampere+ and fp16 on sm_70."""
    amp_dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    if cfg.amp_dtype == "auto":
        cap = torch.cuda.get_device_capability() if cfg.device.startswith("cuda") else (0, 0)
        amp_dtype = torch.bfloat16 if cap >= (8, 0) else torch.float16
    else:
        amp_dtype = amp_dtype_map.get(cfg.amp_dtype, torch.bfloat16)
    amp_enabled = bool(cfg.use_amp and cfg.device.startswith("cuda"))
    return amp_enabled, amp_dtype
