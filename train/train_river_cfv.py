"""Generate exact River DCFR Nash equilibria and train `RiverCFVNet` (ReBeL value net).

Usage:
    uv run python train/train_river_cfv.py \
        --samples 20000 --iters 120 --epochs 25 \
        --device cuda:0 --out checkpoints/river_cfv_v1.pt
"""

from __future__ import annotations

import argparse
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


def _sample_random_range(rng: np.random.Generator, unblocked: np.ndarray) -> np.ndarray:
    """Sample a diverse realistic poker range (mix of wide, polarized, condensed, and tight)."""
    mode = int(rng.integers(0, 4))
    w = np.zeros(len(unblocked), dtype=np.float32)
    if mode == 0:
        # Wide Dirichlet range (limped / checked-down pots).
        w = rng.dirichlet(np.full(len(unblocked), 0.5)).astype(np.float32)
    elif mode == 1:
        # Medium condensed subset (single-raised pot river range: ~25%-60% of combos).
        k = int(rng.integers(max(20, len(unblocked) // 5), max(40, (len(unblocked) * 3) // 5)))
        chosen = rng.choice(len(unblocked), size=k, replace=False)
        w[chosen] = rng.dirichlet(np.ones(k)).astype(np.float32)
    elif mode == 2:
        # Tight 3-bet/4-bet range (~5%-20% of combos).
        k = int(rng.integers(max(12, len(unblocked) // 20), max(30, len(unblocked) // 5)))
        chosen = rng.choice(len(unblocked), size=k, replace=False)
        w[chosen] = rng.dirichlet(np.full(k, 0.7)).astype(np.float32)
    else:
        # Spiky polarized range (log-normal weights).
        raw = np.exp(rng.normal(0.0, 2.0, size=len(unblocked))).astype(np.float32)
        w = raw / max(1e-12, float(raw.sum()))

    full = np.zeros(NUM_COMBOS, dtype=np.float32)
    full[unblocked] = w / max(1e-12, float(w.sum()))
    return full


def generate_dataset(
    *,
    num_samples: int,
    iters: int,
    device: str,
    seed: int = 42,
    bet_fractions: tuple[float, ...] = (0.33, 0.75, 1.5),
    max_raises: int = 2,
) -> dict[str, np.ndarray]:
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
    for i in range(num_samples):
        board = rng.choice(52, size=5, replace=False).tolist()
        unblocked = unblocked_combo_indices(board)
        r0 = _sample_random_range(rng, unblocked)
        r1 = _sample_random_range(rng, unblocked)
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
        pct, valid = river_hand_strength_pct(board, evaluator)

        r0_arr[i] = r0
        r1_arr[i] = r1
        ck0_arr[i] = ck0
        ck1_arr[i] = ck1
        pct_arr[i] = pct
        val_arr[i] = valid
        spr_arr[i] = spr
        y0_arr[i] = cfv0
        y1_arr[i] = cfv1

        if (i + 1) % max(1, num_samples // 10) == 0:
            el = time.perf_counter() - t0
            print(
                f"[cfv data] {i + 1}/{num_samples} ({(i + 1) / el:.1f} solves/s, {el:.1f}s)",
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


def train_cfv(
    data: dict[str, np.ndarray],
    *,
    device: str = "cpu",
    hidden_dim: int = 1024,
    num_blocks: int = 4,
    epochs: int = 20,
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

    tr_sl = slice(0, n_tr)
    va_sl = slice(n_tr, n)

    val_rmse = 0.0
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
            r0v = _to_t("r0", va_sl)
            r1v = _to_t("r1", va_sl)
            p0v, p1v = net(
                r0=r0v,
                r1=r1v,
                check_cfv0=_to_t("ck0", va_sl),
                check_cfv1=_to_t("ck1", va_sl),
                rank_pct=_to_t("pct", va_sl),
                valid_mask=_to_t("valid", va_sl),
                spr=_to_t("spr", va_sl),
            )
            y0v = _to_t("y0", va_sl)
            y1v = _to_t("y1", va_sl)
            vv = _to_t("valid", va_sl)
            mse = (
                0.5
                * (((p0v - y0v).pow(2) * vv).sum() + ((p1v - y1v).pow(2) * vv).sum())
                / vv.sum().clamp_min(1.0)
            )
            val_rmse = float(mse.sqrt().item())
            # Baseline check-down RMSE for comparison.
            ck_mse = (
                0.5
                * (
                    ((_to_t("ck0", va_sl) - y0v).pow(2) * vv).sum()
                    + ((_to_t("ck1", va_sl) - y1v).pow(2) * vv).sum()
                )
                / vv.sum().clamp_min(1.0)
            )
            ck_rmse = float(ck_mse.sqrt().item())
        if (ep + 1) % max(1, epochs // 5) == 0 or ep == epochs - 1:
            print(
                f"[cfv ep {ep + 1:>3}/{epochs}] train_huber={tr_loss_sum / max(1, nb):.5f} "
                f"val_rmse={val_rmse * 100:.2f}% pot (checkdown baseline={ck_rmse * 100:.2f}% pot)",
                flush=True,
            )

    _ = tr_sl
    return net, {"val_rmse_pot": val_rmse}


def main() -> int:
    p = argparse.ArgumentParser(description="Train ReBeL RiverCFVNet on exact River DCFR solves.")
    p.add_argument("--samples", type=int, default=10000)
    p.add_argument("--iters", type=int, default=120)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=1024)
    p.add_argument("--num-blocks", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=Path("checkpoints/river_cfv_v1.pt"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    data = generate_dataset(
        num_samples=args.samples,
        iters=args.iters,
        device=args.device,
        seed=args.seed,
    )
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
    save_cfv_checkpoint(args.out, net, meta={"samples": args.samples, **metrics})
    print(f"[cfv] saved {args.out} (val_rmse={metrics['val_rmse_pot'] * 100:.2f}% pot)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
