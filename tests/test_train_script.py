import os

import pytest
import torch


def test_self_play_gae_flips_bootstrap_for_opponent_perspective():
    from train.train import _compute_gae

    rewards = torch.tensor([[0.0]])
    dones = torch.tensor([[False]])
    values = torch.tensor([[0.0]])
    player_id = torch.tensor([[0]])
    last_value = torch.tensor([10.0])
    last_player_id = torch.tensor([1])

    advantages, returns = _compute_gae(
        rewards=rewards,
        dones=dones,
        values=values,
        last_value=last_value,
        player_id=player_id,
        last_player_id=last_player_id,
        gamma=1.0,
        gae_lambda=1.0,
    )

    assert torch.allclose(advantages, torch.tensor([[-10.0]]))
    assert torch.allclose(returns, torch.tensor([[-10.0]]))


def test_self_play_gae_keeps_bootstrap_sign_for_same_player_perspective():
    from train.train import _compute_gae

    rewards = torch.tensor([[0.0]])
    dones = torch.tensor([[False]])
    values = torch.tensor([[0.0]])
    player_id = torch.tensor([[1]])
    last_value = torch.tensor([10.0])
    last_player_id = torch.tensor([1])

    advantages, returns = _compute_gae(
        rewards=rewards,
        dones=dones,
        values=values,
        last_value=last_value,
        player_id=player_id,
        last_player_id=last_player_id,
        gamma=1.0,
        gae_lambda=1.0,
    )

    assert torch.allclose(advantages, torch.tensor([[10.0]]))
    assert torch.allclose(returns, torch.tensor([[10.0]]))


@pytest.mark.skipif(
    not os.path.exists("src/gpu_poker/lookup_data/hand_ranks.npz"),
    reason="no lookup data",
)
def test_train_script_cpu_one_update_smoke():
    from train.train import TrainConfig, run

    cfg = TrainConfig(
        device="cpu",
        num_envs=32,
        rollout_steps=8,
        total_env_steps=256,
        num_updates=0,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=1.0,
        card_embed_dim=16,
        mlp_dim=32,
        torso_layers=1,
        lstm_hidden=32,
        head_layers=0,
        head_dim=None,
        num_epochs=1,
        minibatch_size=64,
        seed=0,
        log_every_env_steps=1,
        log_every_seconds=1.0,
        checkpoint_dir="checkpoints",
        save_every_env_steps=0,
        resume_from="",
        eval_every=0,
        eval_hands=0,
        eval_max_steps=0,
        league_enabled=False,
        league_eta_selfplay=1.0,
        league_bot_initial_score=0.0,
        league_eval_bots=True,
        league_eval_bots_every=1,
        league_snapshot_pool=0,
        league_snapshot_every_updates=0,
        league_eval_every_updates=0,
        league_eval_hands=0,
        league_eval_subset=0,
        league_eval_deterministic=True,
        league_eval_seat_swap=True,
        league_pfsp_temperature=1.0,
        league_pfsp_epsilon=0.0,
        league_rollout_snapshot_k=0,
        selfplay_eval_enabled=False,
        selfplay_eval_every_updates=0,
        selfplay_eval_hands=0,
        selfplay_eval_subset=0,
        selfplay_eval_deterministic=True,
        selfplay_eval_seat_swap=True,
        timing_sync=False,
        compile_policy=False,
        use_amp=False,
        amp_dtype="bfloat16",
        use_cuda_graph=False,
    )
    result = run(cfg)
    assert result.env_steps == 256
    assert result.overall_sps > 0


@pytest.mark.skipif(
    not os.path.exists("src/gpu_poker/lookup_data/hand_ranks.npz"),
    reason="no lookup data",
)
def test_train_script_cpu_transformer_smoke():
    """End-to-end smoke for the transformer backbone: one full rollout +
    PPO update must finish without NaN/crash, exercising the action-token
    push after every env.step and the packed-state buffer round-trip in
    the PPO replay."""
    from train.train import TrainConfig, run

    cfg = TrainConfig(
        device="cpu",
        num_envs=32,
        rollout_steps=8,
        total_env_steps=256,
        num_updates=0,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=1.0,
        # LSTM fields ignored when backbone=transformer; supply sane values
        # so TrainConfig validates.
        card_embed_dim=16,
        mlp_dim=32,
        torso_layers=1,
        lstm_hidden=32,
        head_layers=0,
        head_dim=None,
        num_epochs=1,
        minibatch_size=64,
        seed=0,
        log_every_env_steps=1,
        log_every_seconds=1.0,
        checkpoint_dir="checkpoints",
        save_every_env_steps=0,
        resume_from="",
        eval_every=0,
        eval_hands=0,
        eval_max_steps=0,
        league_enabled=False,
        league_eta_selfplay=1.0,
        league_bot_initial_score=0.0,
        league_eval_bots=True,
        league_eval_bots_every=1,
        league_snapshot_pool=0,
        league_snapshot_every_updates=0,
        league_eval_every_updates=0,
        league_eval_hands=0,
        league_eval_subset=0,
        league_eval_deterministic=True,
        league_eval_seat_swap=True,
        league_pfsp_temperature=1.0,
        league_pfsp_epsilon=0.0,
        league_rollout_snapshot_k=0,
        selfplay_eval_enabled=False,
        selfplay_eval_every_updates=0,
        selfplay_eval_hands=0,
        selfplay_eval_subset=0,
        selfplay_eval_deterministic=True,
        selfplay_eval_seat_swap=True,
        timing_sync=False,
        compile_policy=False,
        use_amp=False,
        amp_dtype="bfloat16",
        use_cuda_graph=False,
        policy_backbone="transformer",
        transformer_d_model=64,
        transformer_n_heads=4,
        transformer_n_layers=2,
        transformer_ffn_dim=128,
    )
    result = run(cfg)
    assert result.env_steps == 256
    assert result.overall_sps > 0


@pytest.mark.skipif(
    not os.path.exists("src/gpu_poker/lookup_data/hand_ranks.npz"),
    reason="no lookup data",
)
def test_train_script_cpu_league_smoke():
    from train.train import TrainConfig, run

    cfg = TrainConfig(
        device="cpu",
        num_envs=16,
        rollout_steps=32,
        total_env_steps=16 * 32,
        num_updates=0,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=1.0,
        card_embed_dim=16,
        mlp_dim=32,
        torso_layers=1,
        lstm_hidden=32,
        head_layers=0,
        head_dim=None,
        num_epochs=1,
        minibatch_size=128,
        seed=0,
        log_every_env_steps=1,
        log_every_seconds=0.1,
        checkpoint_dir="checkpoints",
        save_every_env_steps=0,
        resume_from="",
        eval_every=0,
        eval_hands=0,
        eval_max_steps=0,
        league_enabled=True,
        league_eta_selfplay=0.0,
        league_bot_initial_score=0.0,
        league_eval_bots=True,
        league_eval_bots_every=1,
        league_snapshot_pool=0,
        league_snapshot_every_updates=0,
        league_eval_every_updates=0,
        league_eval_hands=0,
        league_eval_subset=0,
        league_eval_deterministic=True,
        league_eval_seat_swap=True,
        league_pfsp_temperature=1.0,
        league_pfsp_epsilon=0.0,
        league_rollout_snapshot_k=0,
        selfplay_eval_enabled=False,
        selfplay_eval_every_updates=0,
        selfplay_eval_hands=0,
        selfplay_eval_subset=0,
        selfplay_eval_deterministic=True,
        selfplay_eval_seat_swap=True,
        timing_sync=False,
        compile_policy=False,
        use_amp=False,
        amp_dtype="bfloat16",
        use_cuda_graph=False,
    )
    result = run(cfg)
    assert result.env_steps == 16 * 32
    assert result.opt_steps > 0


def test_train_script_cpu_triad_smoke(tmp_path):
    """End-to-end smoke test for the AlphaStar triad training loop.

    Verifies _run_impl_triad runs a complete (rollout + per-role PPO +
    snapshot trigger evaluation) cycle on CPU without errors. num_envs is
    bumped above 192 (64 * 3 roles) to satisfy the triad's per-role
    minimum-envs validation.
    """
    from train.train import TrainConfig, run

    cfg = TrainConfig(
        device="cpu",
        num_envs=384,
        rollout_steps=4,
        total_env_steps=384 * 4 * 2,  # 2 updates
        num_updates=0,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=1.0,
        card_embed_dim=16,
        mlp_dim=32,
        torso_layers=1,
        lstm_hidden=32,
        head_layers=0,
        head_dim=None,
        num_epochs=1,
        minibatch_size=128,
        seed=0,
        log_every_env_steps=1,
        log_every_seconds=0.1,
        checkpoint_dir=str(tmp_path / "ckpt"),
        save_every_env_steps=0,
        resume_from="",
        eval_every=0,
        eval_hands=0,
        eval_max_steps=0,
        league_enabled=False,
        league_eta_selfplay=0.0,
        league_bot_initial_score=0.0,
        league_eval_bots=False,
        league_eval_bots_every=1,
        league_snapshot_pool=0,
        league_snapshot_every_updates=0,
        league_eval_every_updates=0,
        league_eval_hands=0,
        league_eval_subset=0,
        league_eval_deterministic=True,
        league_eval_seat_swap=True,
        league_pfsp_temperature=1.0,
        league_pfsp_epsilon=0.0,
        league_rollout_snapshot_k=0,
        selfplay_eval_enabled=False,
        selfplay_eval_every_updates=0,
        selfplay_eval_hands=0,
        selfplay_eval_subset=0,
        selfplay_eval_deterministic=True,
        selfplay_eval_seat_swap=True,
        timing_sync=False,
        compile_policy=False,
        use_amp=False,
        amp_dtype="bfloat16",
        use_cuda_graph=False,
        policy_backbone="transformer",
        transformer_d_model=32,
        transformer_n_heads=4,
        transformer_n_layers=1,
        transformer_ffn_dim=64,
        # ---- Triad knobs ----
        triad_enabled=True,
        triad_main_snapshot_every=1,  # snapshot every update so triggers fire
        # Smoke test must NOT depend on the production MLflow server. A
        # separate test (test_train_script_cpu_triad_mlflow_smoke) verifies
        # the integration with an isolated experiment.
        mlflow_enabled=False,
    )
    result = run(cfg)
    assert result.env_steps >= 384 * 4 * 2
    # Confirm at least one role had a non-trivial PPO update
    # (sum of mb_count across all roles is surfaced as opt_steps).
    assert result.opt_steps > 0
    assert result.elapsed_s > 0


def test_train_script_cpu_triad_mlflow_smoke(tmp_path):
    """End-to-end MLflow integration check.

    Runs the triad loop with mlflow_enabled=True against the production
    server but under a dedicated "pokergpu-tests" experiment so production
    runs aren't polluted. Verifies the run gets created, params land,
    snapshot tags fire, and the final_status closes out.

    Skipped if mlflow isn't installed or the network is unreachable;
    the production test_train_script_cpu_triad_smoke covers correctness
    of the training loop independently.
    """
    import pytest

    pytest.importorskip("mlflow")
    import socket
    from urllib.parse import urlparse

    from train.mlflow_config import MLflowNotConfiguredError, configure_mlflow, tracking_uri

    # tracking_uri() loads the gitignored .env and returns the configured
    # server URI; it raises if none is set (fresh clone, no .env). It makes
    # no network call, so it is safe for the reachability probe below.
    try:
        uri = tracking_uri()
    except MLflowNotConfiguredError as e:
        pytest.skip(str(e))
    parsed = urlparse(uri)
    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        socket.create_connection((host, port), timeout=3).close()
    except OSError:
        pytest.skip(f"{host}:{port} unreachable from test env")

    import mlflow
    from train.train import TrainConfig, run

    cfg = TrainConfig(
        device="cpu",
        num_envs=384,
        rollout_steps=4,
        total_env_steps=384 * 4 * 2,
        num_updates=0,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=1.0,
        card_embed_dim=16,
        mlp_dim=32,
        torso_layers=1,
        lstm_hidden=32,
        head_layers=0,
        head_dim=None,
        num_epochs=1,
        minibatch_size=128,
        seed=0,
        log_every_env_steps=1,
        log_every_seconds=0.1,
        checkpoint_dir=str(tmp_path / "ckpt"),
        save_every_env_steps=0,
        resume_from="",
        eval_every=0,
        eval_hands=0,
        eval_max_steps=0,
        league_enabled=False,
        league_eta_selfplay=0.0,
        league_bot_initial_score=0.0,
        league_eval_bots=False,
        league_eval_bots_every=1,
        league_snapshot_pool=0,
        league_snapshot_every_updates=0,
        league_eval_every_updates=0,
        league_eval_hands=0,
        league_eval_subset=0,
        league_eval_deterministic=True,
        league_eval_seat_swap=True,
        league_pfsp_temperature=1.0,
        league_pfsp_epsilon=0.0,
        league_rollout_snapshot_k=0,
        selfplay_eval_enabled=False,
        selfplay_eval_every_updates=0,
        selfplay_eval_hands=0,
        selfplay_eval_subset=0,
        selfplay_eval_deterministic=True,
        selfplay_eval_seat_swap=True,
        timing_sync=False,
        compile_policy=False,
        use_amp=False,
        amp_dtype="bfloat16",
        use_cuda_graph=False,
        policy_backbone="transformer",
        transformer_d_model=32,
        transformer_n_heads=4,
        transformer_n_layers=1,
        transformer_ffn_dim=64,
        triad_enabled=True,
        triad_main_snapshot_every=1,
        mlflow_enabled=True,
        mlflow_experiment="pokergpu-tests",
        mlflow_run_name="pytest-triad-smoke",
        mlflow_tags=["smoke", "triad"],
        mlflow_notes="auto-generated by test_train_script_cpu_triad_mlflow_smoke",
    )
    result = run(cfg)
    assert result.env_steps >= 384 * 4 * 2

    # Verify the run lives on the server with the expected metadata.
    configure_mlflow(experiment="pokergpu-tests")
    runs = mlflow.search_runs(
        experiment_names=["pokergpu-tests"],
        filter_string="tags.mlflow.runName = 'pytest-triad-smoke'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    assert len(runs) == 1, "expected exactly one matching run"
    row = runs.iloc[0]
    assert row["tags.final_status"] == "completed"
    assert row["tags.train_mode"] == "triad"
    assert row["tags.label.smoke"] == "true"
    assert row["tags.label.triad"] == "true"
    # At least one per-role metric must have landed.
    assert "metrics.main/loss" in row.index or "metrics.main/snapshot_count" in row.index
