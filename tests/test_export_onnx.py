"""Tests for train/export_onnx.py covering both backbones.

The slow per-batch parity loop is exercised by validate_parity()'s own
script entrypoint; here we keep tests fast (single export, ~10 batch
parity check) and focused on the contract the JS web client relies on:

  * Both backbones export to a self-contained .onnx with the same
    input/output names and dtypes.
  * The ONNX file carries ``policy_backbone`` + ``state_width`` in
    ``model.metadata_props`` so inference.mjs can dispatch on backbone
    without filename / convention reliance.
  * PyTorch <-> onnxruntime outputs agree within 1e-3.
"""

from __future__ import annotations

import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from train.export_onnx import (
    _build_model_from_cfg,
    _default_arch_cfg,
    export,
    validate_parity,
)


def _exported_metadata(path) -> dict[str, str]:
    import onnx

    proto = onnx.load(str(path))
    return {kv.key: kv.value for kv in proto.metadata_props}


def test_export_lstm_init_random(tmp_path):
    cfg = _default_arch_cfg("lstm")
    model = _build_model_from_cfg(cfg).eval()
    out = tmp_path / "lstm.onnx"
    export(model, out)
    assert out.exists()
    assert out.stat().st_size > 1024  # at least non-trivial

    meta = _exported_metadata(out)
    assert meta["policy_backbone"] == "lstm"
    assert int(meta["state_width"]) == int(model.lstm_hidden)
    assert int(meta["scalar_dim"]) == 17

    # Fast parity check.
    validate_parity(model, out, n_batches=10)


def test_export_transformer_init_random(tmp_path):
    cfg = _default_arch_cfg("transformer")
    # Shrink for test speed -- defaults are full-size for v100.
    cfg.update(
        transformer_d_model=32,
        transformer_n_heads=4,
        transformer_n_layers=1,
        transformer_ffn_dim=64,
    )
    model = _build_model_from_cfg(cfg).eval()
    out = tmp_path / "transformer.onnx"
    export(model, out)
    assert out.exists()

    meta = _exported_metadata(out)
    assert meta["policy_backbone"] == "transformer"
    # Transformer state width is fixed = MAX_TOKENS * TOKEN_FIELDS + 1 = 129.
    assert int(meta["state_width"]) == 129

    # Parity check tolerates the same 1e-3 threshold as the LSTM path.
    validate_parity(model, out, n_batches=10)


def test_export_io_signature_is_backbone_invariant(tmp_path):
    """JS web client relies on identical I/O names for both backbones so
    it can swap models without changing tensor wiring.
    """
    import onnxruntime as ort

    cfg_lstm = _default_arch_cfg("lstm")
    model_l = _build_model_from_cfg(cfg_lstm).eval()
    out_l = tmp_path / "lstm.onnx"
    export(model_l, out_l)

    cfg_t = _default_arch_cfg("transformer")
    cfg_t.update(transformer_d_model=32, transformer_n_layers=1)
    model_t = _build_model_from_cfg(cfg_t).eval()
    out_t = tmp_path / "transformer.onnx"
    export(model_t, out_t)

    sess_l = ort.InferenceSession(str(out_l), providers=["CPUExecutionProvider"])
    sess_t = ort.InferenceSession(str(out_t), providers=["CPUExecutionProvider"])
    assert sess_l.get_inputs() and sess_t.get_inputs()
    assert (
        [i.name for i in sess_l.get_inputs()]
        == [i.name for i in sess_t.get_inputs()]
        == ["cards", "scalars", "action_mask", "h", "c"]
    )
    assert (
        [o.name for o in sess_l.get_outputs()]
        == [o.name for o in sess_t.get_outputs()]
        == ["action_logits", "raise_logits", "value", "h_new", "c_new"]
    )
