from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from train.checkpointing import CheckpointState, load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip_cpu(tmp_path: Path) -> None:
    model = torch.nn.Linear(4, 3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Make weights non-default.
    x = torch.randn(8, 4)
    y = model(x).sum()
    y.backward()
    opt.step()

    p = tmp_path / "ckpt.pt"
    save_checkpoint(
        path=p,
        model=model,
        optimizer=opt,
        state=CheckpointState(env_steps=123, opt_steps=456, update=7),
    )

    model2 = torch.nn.Linear(4, 3)
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    st = load_checkpoint(path=p, model=model2, optimizer=opt2, device=torch.device("cpu"))

    assert st.env_steps == 123
    assert st.opt_steps == 456
    assert st.update == 7

    for a, b in zip(model.parameters(), model2.parameters(), strict=True):
        assert torch.allclose(a, b)


def test_checkpoint_load_accepts_non_tensor_rng_state(tmp_path: Path) -> None:
    model = torch.nn.Linear(4, 3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    p = tmp_path / "ckpt.pt"
    save_checkpoint(
        path=p,
        model=model,
        optimizer=opt,
        state=CheckpointState(env_steps=1, opt_steps=2, update=3),
    )

    payload = torch.load(p, map_location="cpu")
    # Simulate a "bad" / cross-device checkpoint where RNG state isn't a tensor.
    payload["rng"]["torch"] = payload["rng"]["torch"].tolist()
    torch.save(payload, p)

    model2 = torch.nn.Linear(4, 3)
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    st = load_checkpoint(path=p, model=model2, optimizer=opt2, device=torch.device("cpu"))
    assert st.env_steps == 1


@pytest.mark.skipif(
    not os.path.exists("src/gpu_poker/lookup_data/hand_ranks.npz"),
    reason="no lookup data",
)
def test_train_save_and_resume_checkpoint_cpu(tmp_path: Path) -> None:
    # Load the real config file, but do not resolve Hydra interpolations like `${now:...}`.
    from omegaconf import OmegaConf
    from train.train import TrainConfig, run

    base = OmegaConf.to_container(OmegaConf.load("conf/train.yaml"), resolve=False)
    assert isinstance(base, dict)
    base.pop("hydra", None)

    num_envs = 32
    rollout_steps = 8
    env_steps_per_rollout = num_envs * rollout_steps

    ckpt_dir = tmp_path / "checkpoints"
    overrides = {
        "device": "cpu",
        "num_envs": num_envs,
        "rollout_steps": rollout_steps,
        "total_env_steps": env_steps_per_rollout,
        "num_updates": 0,
        "num_epochs": 1,
        "minibatch_size": 64,
        "seed": 0,
        "log_every_env_steps": 1,
        "log_every_seconds": 0.1,
        # Use a tiny policy for speed.
        "card_embed_dim": 16,
        "mlp_dim": 32,
        "torso_layers": 1,
        "lstm_hidden": 32,
        "head_layers": 0,
        "head_dim": None,
        "checkpoint_dir": str(ckpt_dir),
        "save_every_env_steps": env_steps_per_rollout,
        "resume_from": "",
        "eval_every": 0,
        "eval_hands": 0,
        "eval_max_steps": 0,
        # League disabled in this checkpoint/resume test.
        "league_enabled": False,
        "league_eta_selfplay": 1.0,
        "league_bot_initial_score": 0.0,
        "league_eval_bots": False,
        "league_eval_bots_every": 1,
        "league_snapshot_pool": 0,
        "league_snapshot_every_updates": 0,
        "league_eval_every_updates": 0,
        "league_eval_hands": 0,
        "league_eval_subset": 0,
        "league_eval_deterministic": True,
        "league_eval_seat_swap": True,
        "league_pfsp_temperature": 1.0,
        "league_pfsp_epsilon": 0.0,
        "league_rollout_snapshot_k": 0,
        # Self-play eval disabled for this checkpoint/resume test.
        "selfplay_eval_enabled": False,
        "selfplay_eval_every_updates": 0,
        "selfplay_eval_hands": 0,
        "selfplay_eval_subset": 0,
        "selfplay_eval_deterministic": True,
        "selfplay_eval_seat_swap": True,
        "timing_sync": False,
        "compile_policy": False,
        "use_amp": False,
        "amp_dtype": "bfloat16",
        "use_cuda_graph": False,
    }
    base.update(overrides)

    cfg1 = TrainConfig(**base)
    r1 = run(cfg1)
    assert r1.env_steps == env_steps_per_rollout
    assert r1.opt_steps > 0

    ckpts = sorted(ckpt_dir.glob("ckpt_env*.pt"))
    assert ckpts, "expected at least one checkpoint file to be written"
    ckpt_path = ckpts[-1]

    ckpt_payload = torch.load(ckpt_path, map_location="cpu")
    assert ckpt_payload["state"]["env_steps"] == env_steps_per_rollout
    assert ckpt_payload["state"]["opt_steps"] == r1.opt_steps
    assert ckpt_payload["state"]["update"] == 0

    # Resume and run one more rollout.
    base2 = dict(base)
    base2.update(
        {
            "total_env_steps": env_steps_per_rollout * 2,
            "save_every_env_steps": 0,
            "resume_from": str(ckpt_path),
        }
    )
    cfg2 = TrainConfig(**base2)
    r2 = run(cfg2)

    assert r2.env_steps == env_steps_per_rollout * 2
    assert r2.opt_steps > r1.opt_steps
