"""Generate exact River DCFR Nash equilibria and train `RiverCFVNet` (ReBeL value net).

Usage:
    uv run python train/train_river_cfv.py \
        --samples 20000 --iters 120 --workers 4 --epochs 35 \
        --device cuda:0 --out checkpoints/river_cfv_v1.pt
"""

from __future__ import annotations

import argparse
import copy
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as fn

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gpu_poker.cfv_net import (  # noqa: E402
    RiverCFVNet,
    river_hand_strength_pct,
    save_cfv_checkpoint,
)
from gpu_poker.search import (  # noqa: E402
    NUM_COMBOS,
    RiverDCFRSolver,
    RiverHandEvaluator,
    unblocked_combo_indices,
)


def _sample_random_range(
    rng: np.random.Generator,
    unblocked: np.ndarray,
    rank_pct: np.ndarray,
) -> np.ndarray:
    """Sample a realistic river range (mix of wide, tight, polarized, condensed, and linear)."""
    mode = int(rng.integers(0, 7))
    p = rank_pct[unblocked]
    n = len(unblocked)
    w = np.zeros(n, dtype=np.float32)

    if mode == 0:
        # Wide Dirichlet range (limped / checked-down pots).
        w = rng.dirichlet(np.full(n, 0.5)).astype(np.float32)
    elif mode == 1:
        # Medium subset (single-raised pot river range: ~20%-60% of combos).
        k = int(rng.integers(max(20, n // 5), max(40, (n * 3) // 5)))
        chosen = rng.choice(n, size=k, replace=False)
        w[chosen] = rng.dirichlet(np.ones(k)).astype(np.float32)
    elif mode == 2:
        # Tight 3-bet/4-bet range (~5%-20% of combos).
        k = int(rng.integers(max(12, n // 20), max(30, n // 5)))
        chosen = rng.choice(n, size=k, replace=False)
        w[chosen] = rng.dirichlet(np.full(k, 0.7)).astype(np.float32)
    elif mode == 3:
        # Spiky log-normal range.
        raw = np.exp(rng.normal(0.0, 2.0, size=n)).astype(np.float32)
        w = raw / max(1e-12, float(raw.sum()))
    elif mode == 4:
        # Polarized betting range (nuts + bluffs, very few medium bluff-catchers).
        nut_cut = float(rng.uniform(0.75, 0.93))
        bluff_cut = float(rng.uniform(0.08, 0.30))
        bluff_share = float(rng.uniform(0.20, 0.60))
        nuts = (p >= nut_cut).astype(np.float32)
        bluffs = (p <= bluff_cut).astype(np.float32)
        noise = rng.gamma(1.0, 1.0, size=n).astype(np.float32)
        nuts_w = nuts * noise
        bluffs_w = bluffs * noise
        ns = float(nuts_w.sum())
        bs = float(bluffs_w.sum())
        w = (1.0 - bluff_share) * (nuts_w / max(1e-12, ns)) + bluff_share * (
            bluffs_w / max(1e-12, bs)
        )
        # Small residual mass on intermediate hands so beliefs stay smooth.
        w += 0.03 * (noise / max(1e-12, float(noise.sum())))
    elif mode == 5:
        # Condensed / capped calling range (concentrated around medium strength).
        center = float(rng.uniform(0.35, 0.75))
        sigma = float(rng.uniform(0.10, 0.25))
        bell = np.exp(-0.5 * ((p - center) / sigma) ** 2).astype(np.float32)
        noise = rng.gamma(1.0, 1.0, size=n).astype(np.float32)
        w = bell * noise
    else:
        # Linear value-heavy range (exponentially tilted toward stronger hands).
        beta = float(rng.uniform(2.0, 7.0))
        tilt = np.exp(beta * (p - 1.0)).astype(np.float32)
        noise = rng.gamma(1.0, 1.0, size=n).astype(np.float32)
        w = tilt * noise

    full = np.zeros(NUM_COMBOS, dtype=np.float32)
    full[unblocked] = w / max(1e-12, float(w.sum()))
    return full


def _generate_chunk(
    args_tuple: tuple[int, int, str, int, tuple[float, ...], int, int],
) -> dict[str, np.ndarray]:
    num_samples, iters, device, seed, bet_fractions, max_raises, worker_id = args_tuple
    if device == "cpu":
        torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    evaluator = RiverHandEvaluator()
    solver = RiverDCFRSolver(
        bet_fractions=bet_fractions, max_raises=max_raises, evaluator=evaluator, device=device
    )

    r0_arr = np.empty((num_samples, NUM_COMBOS), dtype=np.float32)
    r1_arr = np.empty((num_samples, NUM_COMBOS), dtype=np.float32)
    ck0_arr = np.empty((num_samples, NUM_COMBOS), dtype=np.float32)
    ck1_arr = np.empty((num_samples, NUM_COMBOS), dtype=np.float32)
    pct_arr = np.empty((num_samples, NUM_COMBOS), dtype=np.float32)
    val_arr = np.empty((num_samples, NUM_COMBOS), dtype=np.float32)
    spr_arr = np.empty((num_samples,), dtype=np.float32)
    y0_arr = np.empty((num_samples, NUM_COMBOS), dtype=np.float32)
    y1_arr = np.empty((num_samples, NUM_COMBOS), dtype=np.float32)

    t0 = time.perf_counter()
    log_step = max(1, num_samples // 10)
    for i in range(num_samples):
        board = rng.choice(52, size=5, replace=False).tolist()
        unblocked = unblocked_combo_indices(board)
        pct, valid = river_hand_strength_pct(board, evaluator)
        r0 = _sample_random_range(rng, unblocked, pct)
        r1 = _sample_random_range(rng, unblocked, pct)
        # Log-uniform SPR in [0.1, 12.0] plus 5% exact all-in (SPR=0).
        spr = 0.0 if rng.random() < 0.05 else float(np.exp(rng.uniform(np.log(0.1), np.log(12.0))))
        pot = 1000
        eff_stack = round(spr * pot)

        cfv0, cfv1, ck0, ck1 = solver.solve_cfvs(
            board=board,
            pot=pot,
            eff_stack=eff_stack,
            range_p0=r0,
            range_p1=r1,
            big_blind=50,
            iterations=iters,
            prune_eps=1e-5,
        )

        r0_arr[i] = r0
        r1_arr[i] = r1
        ck0_arr[i] = ck0
        ck1_arr[i] = ck1
        pct_arr[i] = pct
        val_arr[i] = valid
        spr_arr[i] = spr
        y0_arr[i] = cfv0
        y1_arr[i] = cfv1

        if worker_id == 0 and (i + 1) % log_step == 0:
            el = time.perf_counter() - t0
            print(
                f"[cfv data w0] {i + 1}/{num_samples} "
                f"({(i + 1) / el:.1f} solves/s/worker, {el:.1f}s)",
                flush=True,
            )

    return {
        "r0": r0_arr,
        "r1": r1_arr,
        "ck0": ck0_arr,
        "ck1": ck1_arr,
        "pct": pct_arr,
        "valid": val_arr,
        "spr": spr_arr,
        "y0": y0_arr,
        "y1": y1_arr,
    }


def generate_dataset(
    *,
    num_samples: int,
    iters: int,
    device: str,
    seed: int = 42,
    bet_fractions: tuple[float, ...] = (0.33, 0.75, 1.5),
    max_raises: int = 2,
    workers: int = 1,
) -> dict[str, np.ndarray]:
    t0 = time.perf_counter()
    workers = max(1, min(int(workers), num_samples))
    if workers == 1:
        data = _generate_chunk((num_samples, iters, device, seed, bet_fractions, max_raises, 0))
    else:
        base = num_samples // workers
        rem = num_samples % workers
        tasks = [
            (
                base + (1 if w < rem else 0),
                iters,
                device,
                seed + 10007 * w,
                bet_fractions,
                max_raises,
                w,
            )
            for w in range(workers)
        ]
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            chunks = pool.map(_generate_chunk, tasks)
        data = {k: np.concatenate([c[k] for c in chunks], axis=0) for k in chunks[0]}
    el = time.perf_counter() - t0
    print(
        f"[cfv data] generated {num_samples} exact Nash river solves in {el:.1f}s "
        f"({num_samples / max(1e-6, el):.1f} solves/s, workers={workers}, device={device})",
        flush=True,
    )
    return data


def train_cfv(
    data: dict[str, np.ndarray],
    *,
    device: str = "cpu",
    hidden_dim: int = 64,
    num_blocks: int = 2,
    epochs: int = 35,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 42,
) -> tuple[RiverCFVNet, dict[str, float]]:
    torch.manual_seed(seed)
    dev = torch.device(device)
    n = data["spr"].shape[0]
    n_val = max(1, min(n // 10, 2000))
    n_tr = n - n_val

    net = RiverCFVNet(hidden_dim=hidden_dim, num_blocks=num_blocks).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    def _to_t(k: str, sl: slice) -> torch.Tensor:
        return torch.from_numpy(data[k][sl]).to(dev)

    va_sl = slice(n_tr, n)
    r0v = _to_t("r0", va_sl)
    r1v = _to_t("r1", va_sl)
    ck0v = _to_t("ck0", va_sl)
    ck1v = _to_t("ck1", va_sl)
    pctv = _to_t("pct", va_sl)
    vv = _to_t("valid", va_sl)
    sprv = _to_t("spr", va_sl)
    y0v = _to_t("y0", va_sl)
    y1v = _to_t("y1", va_sl)

    with torch.no_grad():
        ck_mse = (
            0.5
            * (((ck0v - y0v).pow(2) * vv).sum() + ((ck1v - y1v).pow(2) * vv).sum())
            / vv.sum().clamp_min(1.0)
        )
        ck_rmse = float(ck_mse.sqrt().item())
        ck_wmse = (
            0.5
            * (((ck0v - y0v).pow(2) * r0v).sum() + ((ck1v - y1v).pow(2) * r1v).sum())
            / (0.5 * (r0v.sum() + r1v.sum())).clamp_min(1e-6)
        )
        ck_wrmse = float(ck_wmse.sqrt().item())

    best_wrmse = float("inf")
    best_rmse = float("inf")
    best_state = copy.deepcopy(net.state_dict())

    for ep in range(epochs):
        net.train()
        perm = np.random.default_rng(seed + ep).permutation(n_tr)
        tr_loss_sum = 0.0
        nb = 0
        for s in range(0, n_tr, batch_size):
            idx = perm[s : s + batch_size]
            r0 = torch.from_numpy(data["r0"][idx]).to(dev)
            r1 = torch.from_numpy(data["r1"][idx]).to(dev)
            ck0 = torch.from_numpy(data["ck0"][idx]).to(dev)
            ck1 = torch.from_numpy(data["ck1"][idx]).to(dev)
            pct = torch.from_numpy(data["pct"][idx]).to(dev)
            valid = torch.from_numpy(data["valid"][idx]).to(dev)
            spr = torch.from_numpy(data["spr"][idx]).to(dev)
            y0 = torch.from_numpy(data["y0"][idx]).to(dev)
            y1 = torch.from_numpy(data["y1"][idx]).to(dev)

            p0, p1 = net(
                r0=r0,
                r1=r1,
                check_cfv0=ck0,
                check_cfv1=ck1,
                rank_pct=pct,
                valid_mask=valid,
                spr=spr,
            )
            w0 = (0.5 * r0 * 1081.0 + 0.5) * valid
            w1 = (0.5 * r1 * 1081.0 + 0.5) * valid
            loss0 = (
                w0 * fn.huber_loss(p0, y0, reduction="none", delta=0.1)
            ).sum() / w0.sum().clamp_min(1.0)
            loss1 = (
                w1 * fn.huber_loss(p1, y1, reduction="none", delta=0.1)
            ).sum() / w1.sum().clamp_min(1.0)
            loss = 0.5 * (loss0 + loss1)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tr_loss_sum += float(loss.item())
            nb += 1

        sched.step()

        net.eval()
        with torch.no_grad():
            p0v, p1v = net(
                r0=r0v,
                r1=r1v,
                check_cfv0=ck0v,
                check_cfv1=ck1v,
                rank_pct=pctv,
                valid_mask=vv,
                spr=sprv,
            )
            mse = (
                0.5
                * (((p0v - y0v).pow(2) * vv).sum() + ((p1v - y1v).pow(2) * vv).sum())
                / vv.sum().clamp_min(1.0)
            )
            val_rmse = float(mse.sqrt().item())
            wmse = (
                0.5
                * (((p0v - y0v).pow(2) * r0v).sum() + ((p1v - y1v).pow(2) * r1v).sum())
                / (0.5 * (r0v.sum() + r1v.sum())).clamp_min(1e-6)
            )
            val_wrmse = float(wmse.sqrt().item())

        if val_wrmse < best_wrmse:
            best_wrmse = val_wrmse
            best_rmse = val_rmse
            best_state = copy.deepcopy(net.state_dict())

        if (ep + 1) % max(1, epochs // 7) == 0 or ep == epochs - 1:
            print(
                f"[cfv ep {ep + 1:>3}/{epochs}] train_huber={tr_loss_sum / max(1, nb):.5f} "
                f"val_rmse={val_rmse * 100:.2f}% (ck={ck_rmse * 100:.2f}%) "
                f"val_wrmse={val_wrmse * 100:.2f}% (ck={ck_wrmse * 100:.2f}%)",
                flush=True,
            )

    net.load_state_dict(best_state)
    return net, {
        "val_rmse_pot": best_rmse,
        "val_wrmse_pot": best_wrmse,
        "checkdown_rmse_pot": ck_rmse,
        "checkdown_wrmse_pot": ck_wrmse,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Train ReBeL RiverCFVNet on exact River DCFR solves.")
    p.add_argument("--samples", type=int, default=15000)
    p.add_argument("--iters", type=int, default=120)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--gen-device",
        type=str,
        default="cpu",
        help="Device for DCFR dataset generation ('cpu' recommended with --workers > 1).",
    )
    p.add_argument(
        "--bet-fractions",
        type=str,
        default="0.5,1.0",
        help="Comma-separated river bet fractions (plus all-in) used for exact Nash solves.",
    )
    p.add_argument("--save-data", type=Path, default=None)
    p.add_argument("--load-data", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("checkpoints/river_cfv_v1.pt"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    bet_fracs = tuple(float(x.strip()) for x in args.bet_fractions.split(",") if x.strip())
    if args.load_data is not None:
        npz = np.load(args.load_data)
        data = {k: npz[k] for k in npz.files}
        print(f"[cfv data] loaded {data['spr'].shape[0]} solves from {args.load_data}", flush=True)
    else:
        data = generate_dataset(
            num_samples=args.samples,
            iters=args.iters,
            device=args.gen_device,
            seed=args.seed,
            bet_fractions=bet_fracs,
            workers=args.workers,
        )
        if args.save_data is not None:
            args.save_data.parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.save_data, **data)
            print(f"[cfv data] saved dataset to {args.save_data}", flush=True)

    net, metrics = train_cfv(
        data,
        device=args.device,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
    save_cfv_checkpoint(args.out, net, meta={"samples": int(data["spr"].shape[0]), **metrics})
    print(
        f"[cfv] saved {args.out} "
        f"(val_rmse={metrics['val_rmse_pot'] * 100:.2f}% pot, "
        f"val_wrmse={metrics['val_wrmse_pot'] * 100:.2f}% pot, "
        f"checkdown_wrmse={metrics['checkdown_wrmse_pot'] * 100:.2f}% pot)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
