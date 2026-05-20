from __future__ import annotations

from pathlib import Path

from train.query_training import _find_latest_hydra_run, _find_tensorboard_dir, _resolve_run_paths


def test_find_latest_hydra_run(tmp_path: Path) -> None:
    outputs = tmp_path / "train" / "outputs"
    a = outputs / "2025-01-01" / "00-00-00"
    b = outputs / "2025-01-01" / "00-00-01"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    (a / "results.json").write_text("{}\n")
    (b / "results.json").write_text("{}\n")

    latest = _find_latest_hydra_run(outputs_dir=outputs)
    assert latest == b


def test_find_latest_hydra_run_works_with_tb_only(tmp_path: Path) -> None:
    outputs = tmp_path / "train" / "outputs"
    run_dir = outputs / "2025-01-01" / "00-00-00"
    tb_dir = run_dir / "train" / "tensorboard"
    tb_dir.mkdir(parents=True)
    (tb_dir / "events.out.tfevents.0").write_text("")

    latest = _find_latest_hydra_run(outputs_dir=outputs)
    assert latest == run_dir


def test_find_tensorboard_dir_prefers_known_locations(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "train" / "tensorboard").mkdir(parents=True)
    (run_dir / "train" / "tensorboard" / "events.out.tfevents.0").write_text("")

    tb = _find_tensorboard_dir(run_dir)
    assert tb == run_dir / "train" / "tensorboard"


def test_resolve_run_paths_falls_back_to_latest_train_json(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path
    (repo_root / "pyproject.toml").write_text("[project]\nname='x'\nversion='0'\n")
    (repo_root / "train").mkdir()
    (repo_root / "train" / "latest_train.json").write_text("{}\n")

    monkeypatch.chdir(repo_root)
    paths = _resolve_run_paths(None)
    assert paths.results_json is not None
    assert paths.results_json.name == "latest_train.json"
