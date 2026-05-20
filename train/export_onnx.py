"""
Export a trained PokerPolicyNet checkpoint to ONNX for browser/Node inference.

The exported graph is **just the deterministic network**: it consumes
(cards, scalars, action_mask, h, c) and returns raw `action_logits`, raw
`raise_params`, `value`, and the new LSTM state `h_new, c_new`.

All distribution math (masking, Categorical sampling, Beta mixture handling)
is left to the consumer (JS in the browser). This avoids exporting
non-trivial ops like `Categorical.sample` / `Beta.sample` that don't have
clean ONNX equivalents, and keeps the graph small and portable.

Usage:
    uv run python train/export_onnx.py \
        --checkpoint train/outputs/2026-05-16/10-36-04/global_league/checkpoints/ckpt_xxx.pt \
        --out web/model.onnx

    # No checkpoint (fresh-init smoke test of the export pipeline):
    uv run python train/export_onnx.py --init-random --out /tmp/model.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gpu_poker import constants as const  # noqa: E402
from gpu_poker.policy import PokerPolicyNet  # noqa: E402
from gpu_poker.policy_transformer import (  # noqa: E402
    STATE_WIDTH as TRANSFORMER_STATE_WIDTH,
)
from gpu_poker.policy_transformer import (  # noqa: E402
    PokerTransformerPolicyNet,
)

# Matches the obs shape produced by WarpPokerEnv.
_SCALAR_DIM = 17
_NUM_CARDS = 7


class ExportablePolicy(nn.Module):
    """Thin LSTM wrapper that strips out distribution sampling for clean ONNX export."""

    def __init__(self, net: PokerPolicyNet):
        super().__init__()
        self.net = net

    def forward(
        self,
        cards: torch.Tensor,  # [B, 7] int64
        scalars: torch.Tensor,  # [B, S] float32
        action_mask: torch.Tensor,  # [B, 4] bool
        h: torch.Tensor,  # [1, B, H] float32
        c: torch.Tensor,  # [1, B, H] float32
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Mirrors PokerPolicyNet._forward_latent up through the heads, but
        # returns raw logits/params and skips Categorical/Beta construction.
        cards_idx = (cards + 1).clamp(0, 52)
        card_emb = self.net.card_embed(cards_idx)  # [B, 7, E]
        card_feat = card_emb.reshape(card_emb.shape[0], -1)  # [B, 7E]

        scal_feat = self.net.scalar_mlp(scalars)  # [B, M]
        x = torch.cat([card_feat, scal_feat], dim=-1)  # [B, 7E+M]
        x = self.net.pre_lstm(x).unsqueeze(1)  # [B, 1, M]

        y, (h_new, c_new) = self.net.lstm(x, (h, c))  # y: [B, 1, H]
        y = y.squeeze(1)  # [B, H]
        y = self.net.post_lstm(y)  # [B, head_dim]

        # Apply action mask: invalid actions get -inf so JS softmax zeros them.
        # We do the masking here (rather than in JS) so the consumer just calls
        # softmax/argmax. action_mask defensive fold-fallback is also baked in.
        logits = self.net.action_head(y)  # [B, 4]
        mask = action_mask.to(dtype=torch.bool)
        mask_sum = mask.sum(dim=-1, keepdim=True)
        fold_col = mask[:, const.ACTION_FOLD : const.ACTION_FOLD + 1] | (mask_sum == 0)
        mask = torch.cat(
            [
                mask[:, : const.ACTION_FOLD],
                fold_col,
                mask[:, const.ACTION_FOLD + 1 :],
            ],
            dim=-1,
        )
        neg_inf = torch.full_like(logits, float(torch.finfo(logits.dtype).min))
        masked_logits = torch.where(mask, logits, neg_inf)

        raise_logits = self.net.raise_head(y)  # [B, NUM_RAISE_BUCKETS]
        value = self.net.value_head(y).squeeze(-1)  # [B]

        return masked_logits, raise_logits, value, h_new, c_new


class ExportableTransformerPolicy(nn.Module):
    """Thin transformer wrapper with the SAME I/O signature as ExportablePolicy.

    Keeping inputs/outputs identical to the LSTM export means the JS web
    client can swap models without changing its tensor wiring. The only
    semantic differences for the JS consumer:

      * ``h`` is the packed transformer state ``[1, B, STATE_WIDTH=129]``
        fp32 (token table + length packed), NOT an LSTM hidden state.
      * ``c`` is unused; we pass it through to keep the signature
        symmetric and let the JS reuse its zero-buffer allocator.
      * ``h_new`` is passthrough -- this graph does the encoder forward
        only. The JS client must call ``pushActionToken(h, ...)`` (mirror
        of ``push_action_token_packed``) AFTER it samples the action to
        record that action in the per-env token table for the next call.

    All downstream sampling math (softmax, raise-bucket -> amount) stays
    in JS exactly as for the LSTM model.
    """

    def __init__(self, net: PokerTransformerPolicyNet):
        super().__init__()
        self.net = net

    def forward(
        self,
        cards: torch.Tensor,  # [B, 7] int64
        scalars: torch.Tensor,  # [B, S] float32
        action_mask: torch.Tensor,  # [B, 4] bool
        h: torch.Tensor,  # [1, B, STATE_WIDTH] float32 (packed)
        c: torch.Tensor,  # [1, B, STATE_WIDTH] float32 (unused, passthrough)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Unpack the per-env token table + length from the float32 packed
        # state. Round-trip is exact at our small int vocab sizes.
        tokens, lengths = self.net.unpack_state(h)

        # Mirror _forward_latent without the Categorical wrappers (those
        # don't export cleanly to ONNX -- sampling stays in JS).
        hist_emb = self.net._embed_history(tokens)  # [B, MAX_TOKENS, D]
        cur_tok = self.net._build_current_token(cards, scalars).unsqueeze(1)
        x = torch.cat([cur_tok, hist_emb], dim=1)
        x = x + self.net.pos_enc.to(x.dtype).unsqueeze(0)

        batch_size = cards.shape[0]
        device = cards.device
        pos = torch.arange(self.net.max_tokens, device=device).unsqueeze(0)
        hist_pad = pos >= lengths.to(dtype=torch.int64).unsqueeze(1)
        cur_pad = torch.zeros((batch_size, 1), device=device, dtype=torch.bool)
        key_padding_mask = torch.cat([cur_pad, hist_pad], dim=1)

        y = self.net.encoder(x, mask=None, src_key_padding_mask=key_padding_mask)
        last = y[:, 0, :]  # [B, D] -- output at the current-state token

        # Same action-masking logic as the LSTM wrapper: ensure FOLD is
        # always available as a fallback when no action is masked in.
        logits = self.net.action_head(last)
        mask = action_mask.to(dtype=torch.bool)
        mask_sum = mask.sum(dim=-1, keepdim=True)
        fold_col = mask[:, const.ACTION_FOLD : const.ACTION_FOLD + 1] | (mask_sum == 0)
        mask = torch.cat(
            [
                mask[:, : const.ACTION_FOLD],
                fold_col,
                mask[:, const.ACTION_FOLD + 1 :],
            ],
            dim=-1,
        )
        neg_inf = torch.full_like(logits, float(torch.finfo(logits.dtype).min))
        masked_logits = torch.where(mask, logits, neg_inf)

        raise_logits = self.net.raise_head(last)
        value = self.net.value_head(last).squeeze(-1)

        # h_new = h passthrough. JS pushes the action token after sampling
        # via pushActionToken() in inference.mjs.
        return masked_logits, raise_logits, value, h, c


def _build_model_from_cfg(cfg: dict) -> nn.Module:
    backbone = str(cfg.get("policy_backbone") or "lstm").lower()
    if backbone == "lstm":
        head_dim = cfg.get("head_dim")
        return PokerPolicyNet(
            scalar_dim=_SCALAR_DIM,
            card_embed_dim=int(cfg["card_embed_dim"]),
            mlp_dim=int(cfg["mlp_dim"]),
            torso_layers=int(cfg["torso_layers"]),
            lstm_hidden=int(cfg["lstm_hidden"]),
            head_layers=int(cfg["head_layers"]),
            head_dim=int(head_dim) if head_dim is not None else None,
        )
    if backbone == "transformer":
        return PokerTransformerPolicyNet(
            scalar_dim=_SCALAR_DIM,
            card_embed_dim=int(cfg.get("card_embed_dim", 32)),
            d_model=int(cfg.get("transformer_d_model", 256)),
            n_heads=int(cfg.get("transformer_n_heads", 4)),
            n_layers=int(cfg.get("transformer_n_layers", 4)),
            ffn_dim=int(cfg.get("transformer_ffn_dim", 1024)),
        )
    raise ValueError(f"Unknown policy_backbone: {backbone!r}")


def _backbone_of(model: nn.Module) -> str:
    """Backbone tag derived from the model class (used for dispatch + ONNX metadata)."""
    if isinstance(model, PokerTransformerPolicyNet):
        return "transformer"
    if isinstance(model, PokerPolicyNet):
        return "lstm"
    raise ValueError(f"Unknown model type for export: {type(model).__name__}")


def _state_width_of(model: nn.Module) -> int:
    """Trailing dim of the h state tensor for this backbone."""
    if isinstance(model, PokerTransformerPolicyNet):
        return TRANSFORMER_STATE_WIDTH
    if isinstance(model, PokerPolicyNet):
        return int(model.lstm_hidden)
    raise ValueError(f"Unknown model type: {type(model).__name__}")


def _wrap_for_export(model: nn.Module) -> nn.Module:
    if isinstance(model, PokerTransformerPolicyNet):
        return ExportableTransformerPolicy(model)
    if isinstance(model, PokerPolicyNet):
        return ExportablePolicy(model)
    raise ValueError(f"Unknown model type: {type(model).__name__}")


def _default_arch_cfg(backbone: str = "lstm") -> dict:
    """Defaults matching conf/train.yaml at the time of writing.

    Only used when --init-random is passed (no real checkpoint).
    """
    if backbone == "transformer":
        return {
            "policy_backbone": "transformer",
            "transformer_d_model": 256,
            "transformer_n_heads": 4,
            "transformer_n_layers": 4,
            "transformer_ffn_dim": 1024,
        }
    return {
        "policy_backbone": "lstm",
        "card_embed_dim": 96,
        "mlp_dim": 384,
        "torso_layers": 4,
        "lstm_hidden": 384,
        "head_layers": 3,
        "head_dim": None,
    }


def _load_arch_cfg(checkpoint_path: Path, override: Path | None) -> dict:
    if override is not None:
        return OmegaConf.to_container(OmegaConf.load(override), resolve=True)  # type: ignore[return-value]
    # Hydra dumps the resolved config to `<run_dir>/.hydra/config.yaml`.
    # Checkpoints typically live at `<run_dir>/<checkpoint_dir>/ckpt_xxx.pt`
    # (e.g. `<run_dir>/global_league/checkpoints/ckpt_xxx.pt`).
    candidates = [
        checkpoint_path.parent / ".hydra" / "config.yaml",  # <run_dir>/.hydra
        checkpoint_path.parent.parent / ".hydra" / "config.yaml",  # one up
        checkpoint_path.parent.parent.parent / ".hydra" / "config.yaml",  # two up (typical)
    ]
    for path in candidates:
        if path.is_file():
            return OmegaConf.to_container(OmegaConf.load(path), resolve=True)  # type: ignore[return-value]
    raise FileNotFoundError(
        f"Could not auto-locate hydra config near checkpoint {checkpoint_path}. "
        f"Tried: {[str(p) for p in candidates]}. Pass --config explicitly."
    )


def _make_dummy_inputs(
    batch: int, state_width: int, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a synthetic (cards, scalars, action_mask, h, c) tuple.

    ``state_width`` is the LSTM hidden size for the lstm backbone OR
    ``STATE_WIDTH`` (==129) for the transformer backbone -- both produce
    a [1, B, state_width] fp32 state tensor and a matching c tensor.
    All-zeros is a valid initial state for both backbones (zero LSTM
    hidden / zero packed transformer state with length=0).
    """
    g = torch.Generator().manual_seed(seed)
    cards = torch.randint(-1, 52, (batch, _NUM_CARDS), generator=g, dtype=torch.int64)
    scalars = torch.randn(batch, _SCALAR_DIM, generator=g, dtype=torch.float32)
    action_mask = torch.zeros(batch, const.NUM_ACTIONS, dtype=torch.bool)
    # Ensure at least one action is legal per row (fold by default + one random).
    action_mask[:, const.ACTION_FOLD] = True
    extra = torch.randint(0, const.NUM_ACTIONS, (batch,), generator=g)
    action_mask[torch.arange(batch), extra] = True
    h = torch.zeros(1, batch, state_width, dtype=torch.float32)
    c_state = torch.zeros(1, batch, state_width, dtype=torch.float32)
    return cards, scalars, action_mask, h, c_state


def export(
    model: nn.Module,
    out_path: Path,
    opset: int = 17,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = _wrap_for_export(model).eval()
    backbone = _backbone_of(model)
    state_width = _state_width_of(model)

    # Tiny representative batch (the actual batch is dynamic at runtime).
    cards, scalars, action_mask, h, c_state = _make_dummy_inputs(batch=4, state_width=state_width)

    # Use `dynamo=False` (the legacy exporter). The new dynamo exporter has a
    # known LSTM bug where it declares h/c outputs as 4-D ([1, 1, B, H]) while
    # producing 3-D outputs ([1, B, H]) — Python onnxruntime warns and runs
    # anyway, but onnxruntime-web/wasm fails strictly at session creation.
    torch.onnx.export(
        wrapper,
        (cards, scalars, action_mask, h, c_state),
        str(out_path),
        input_names=["cards", "scalars", "action_mask", "h", "c"],
        output_names=["action_logits", "raise_logits", "value", "h_new", "c_new"],
        dynamic_axes={
            "cards": {0: "B"},
            "scalars": {0: "B"},
            "action_mask": {0: "B"},
            "h": {1: "B"},
            "c": {1: "B"},
            "action_logits": {0: "B"},
            "raise_logits": {0: "B"},
            "value": {0: "B"},
            "h_new": {1: "B"},
            "c_new": {1: "B"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )

    # Inline external weights AND tag the model with backbone metadata so
    # the JS client knows whether h_new is post-LSTM state (lstm) or a
    # passthrough requiring a JS pushActionToken call (transformer).
    import onnx  # type: ignore[import-untyped]

    model_proto = onnx.load(str(out_path), load_external_data=True)
    _set_metadata(
        model_proto,
        {
            "policy_backbone": backbone,
            "state_width": str(state_width),
            "scalar_dim": str(_SCALAR_DIM),
            "num_cards": str(_NUM_CARDS),
            "num_actions": str(const.NUM_ACTIONS),
        },
    )
    data_sidecar = out_path.with_name(out_path.name + ".data")
    onnx.save_model(model_proto, str(out_path), save_as_external_data=False)
    if data_sidecar.exists():
        data_sidecar.unlink()


def _set_metadata(model_proto, props: dict[str, str]) -> None:
    """Write key/value pairs to ONNX model.metadata_props (overwrites existing
    keys, leaves others untouched). The JS client reads these to dispatch
    on backbone without filename / convention reliance.
    """
    existing = {kv.key: kv for kv in model_proto.metadata_props}
    for key, value in props.items():
        if key in existing:
            existing[key].value = value
        else:
            kv = model_proto.metadata_props.add()
            kv.key = key
            kv.value = value


def validate_parity(model: nn.Module, onnx_path: Path, n_batches: int = 100) -> None:
    """Run N random inputs through PyTorch + onnxruntime, assert outputs match."""
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]
    except ImportError as e:
        raise RuntimeError(
            "onnxruntime not installed. Add it with: uv add --dev onnxruntime"
        ) from e

    wrapper = _wrap_for_export(model).eval()
    state_width = _state_width_of(model)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    max_diffs = dict.fromkeys(["action_logits", "raise_logits", "value", "h_new", "c_new"], 0.0)
    rng = np.random.default_rng(seed=0)

    for i in range(n_batches):
        batch = int(rng.integers(1, 16))
        cards, scalars, action_mask, h, c_state = _make_dummy_inputs(
            batch=batch, state_width=state_width, seed=i + 1
        )

        with torch.no_grad():
            py_out = wrapper(cards, scalars, action_mask, h, c_state)

        onnx_out = sess.run(
            None,
            {
                "cards": cards.numpy(),
                "scalars": scalars.numpy(),
                "action_mask": action_mask.numpy(),
                "h": h.numpy(),
                "c": c_state.numpy(),
            },
        )

        # Replace any masked -inf with a finite sentinel before computing diff
        # (subtracting inf - inf would yield NaN even when both are masked).
        def _finite_for_diff(a: torch.Tensor | np.ndarray) -> np.ndarray:
            arr = a.numpy() if isinstance(a, torch.Tensor) else a
            return np.where(np.isfinite(arr), arr, 0.0)

        for name, py_t, onnx_t in zip(
            ["action_logits", "raise_logits", "value", "h_new", "c_new"],
            py_out,
            onnx_out,
            strict=True,
        ):
            diff = float(np.max(np.abs(_finite_for_diff(py_t) - _finite_for_diff(onnx_t))))
            max_diffs[name] = max(max_diffs[name], diff)

    print(f"[validate] max abs diff PyTorch vs ONNX across {n_batches} batches:")
    for name, d in max_diffs.items():
        flag = "OK" if d < 1e-3 else "WARN" if d < 1e-1 else "FAIL"
        print(f"  {name:15s} = {d:>12.4e}   [{flag}]")

    worst = max(max_diffs.values())
    if worst > 1e-3:
        raise SystemExit(f"Parity check FAILED: worst diff {worst:.4e} > 1e-3")
    print(f"[validate] PASS (worst diff {worst:.4e})")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, help="Path to a .pt checkpoint.")
    p.add_argument(
        "--config",
        type=Path,
        help="Path to hydra config.yaml. Auto-detected from checkpoint location if omitted.",
    )
    p.add_argument("--out", type=Path, required=True, help="Output .onnx path.")
    p.add_argument(
        "--init-random",
        action="store_true",
        help="Skip checkpoint loading; export a fresh-init model (smoke test only).",
    )
    p.add_argument(
        "--backbone",
        choices=["lstm", "transformer"],
        default="lstm",
        help="Only used with --init-random; picks which fresh-init arch to build.",
    )
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--no-validate", action="store_true", help="Skip Python↔ONNX parity check.")
    args = p.parse_args()

    if args.init_random:
        cfg = _default_arch_cfg(args.backbone)
        print(f"[export] init-random with arch={cfg}")
        model = _build_model_from_cfg(cfg)
    else:
        if args.checkpoint is None:
            p.error("Either --checkpoint PATH or --init-random must be provided.")
        cfg = _load_arch_cfg(args.checkpoint, args.config)
        backbone = str(cfg.get("policy_backbone") or "lstm").lower()
        if backbone == "transformer":
            print(
                f"[export] arch from config (transformer): "
                f"d_model={cfg.get('transformer_d_model')} "
                f"n_heads={cfg.get('transformer_n_heads')} "
                f"n_layers={cfg.get('transformer_n_layers')} "
                f"ffn={cfg.get('transformer_ffn_dim')}"
            )
        else:
            print(
                f"[export] arch from config (lstm): card={cfg['card_embed_dim']} "
                f"mlp={cfg['mlp_dim']} torso={cfg['torso_layers']} "
                f"lstm={cfg['lstm_hidden']} head_layers={cfg['head_layers']}"
            )
        model = _build_model_from_cfg(cfg)
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        # torch.compile() wraps the module; saved state_dict gets keys prefixed
        # with "_orig_mod." that won't match a fresh (uncompiled) PokerPolicyNet.
        # Strip that prefix so we can load into the plain model used for export.
        sd = payload["model"]
        if any(k.startswith("_orig_mod.") for k in sd):
            sd = {k[len("_orig_mod.") :]: v for k, v in sd.items()}
        model.load_state_dict(sd)
        print(f"[export] loaded weights from {args.checkpoint}")

    model.eval()

    export(model, args.out, opset=args.opset)
    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"[export] wrote {args.out} ({size_mb:.2f} MB)")

    if not args.no_validate:
        validate_parity(model, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
