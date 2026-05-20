// ONNX load + forward test. Mirrors what the browser does in
// src/inference.mjs::loadModel for a File / Blob input. Used as a CI
// catch for export-side breakages that produce ONNX files the
// onnxruntime-web runtime can't parse (e.g. the "protobuf parsing
// failed" error users see in the UI).
//
// The fixture is checked in as tests/fixtures.onnx -- regenerate via:
//
//     uv run python train/export_onnx.py --checkpoint X.pt \
//       --config /tmp/triad_arch.yaml --out web/tests/fixtures.onnx
//
// If you change the export format or upgrade onnxruntime-web, run this
// test FIRST to catch the regression before users hit the drag-and-drop
// failure in production.

import { test } from "node:test";
import { strict as assert } from "node:assert";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import * as ort from "onnxruntime-web";

const __dirname = dirname(fileURLToPath(import.meta.url));
const FIXTURE_PATH = resolve(__dirname, "fixtures.onnx");

// Triad-spec defaults; mirror exporter output.
const NUM_CARDS = 7;
const SCALAR_DIM = 17;
const NUM_ACTIONS = 4;
const TRANSFORMER_STATE_WIDTH = 32 * 4 + 1; // 129

test("onnxruntime-web parses our exported .onnx (regression for protobuf-parsing-failed)", async () => {
  const bytes = await readFile(FIXTURE_PATH);

  // Match src/inference.mjs exactly: single-thread WASM, no SAB.
  ort.env.wasm.numThreads = 1;

  const session = await ort.InferenceSession.create(bytes, {
    executionProviders: ["wasm"],
  });

  // I/O signature must match what inference.mjs expects.
  for (const name of ["cards", "scalars", "action_mask", "h", "c"]) {
    assert.ok(
      session.inputNames.includes(name),
      `missing input '${name}'; got [${session.inputNames.join(", ")}]`,
    );
  }
  for (const name of ["action_logits", "raise_logits", "value", "h_new", "c_new"]) {
    assert.ok(
      session.outputNames.includes(name),
      `missing output '${name}'; got [${session.outputNames.join(", ")}]`,
    );
  }

  // Input `h` shape must declare the state width so inference.mjs can
  // size hidden tensors correctly. With the transformer backbone the
  // last dim of h is 129 (32 tokens x 4 fields + length); with LSTM it's
  // the lstm_hidden config value. We previously relied on metadata_props
  // (modelMetadata.customMetadataMap) but onnxruntime-web 1.20+ doesn't
  // expose that, so the input-shape route is the durable contract.
  const hMeta = session.inputMetadata?.h ?? session.inputMetadata?.[3];
  assert.ok(hMeta, "missing inputMetadata for 'h'");
  assert.ok(Array.isArray(hMeta.shape), "h.shape not an array");
  const hWidth = hMeta.shape[2];
  assert.equal(
    typeof hWidth,
    "number",
    `h.shape[2] must be a concrete number, got ${JSON.stringify(hWidth)}`,
  );
  assert.ok(
    hWidth === TRANSFORMER_STATE_WIDTH || hWidth >= 64,
    `h.shape[2]=${hWidth} not a recognized state width`,
  );
});

test("session can run one forward pass with dummy inputs", async () => {
  const bytes = await readFile(FIXTURE_PATH);
  ort.env.wasm.numThreads = 1;
  const session = await ort.InferenceSession.create(bytes, {
    executionProviders: ["wasm"],
  });

  // Mirror inference.mjs detection (input-shape based, since modelMetadata
  // isn't exposed by onnxruntime-web).
  const hMeta = session.inputMetadata?.h ?? session.inputMetadata?.[3];
  const stateWidth = (hMeta && Array.isArray(hMeta.shape) && typeof hMeta.shape[2] === "number")
    ? hMeta.shape[2]
    : TRANSFORMER_STATE_WIDTH;

  // Dummy single-batch obs. Cards as int64, scalars as float32, action_mask as bool.
  const cards = BigInt64Array.from({ length: NUM_CARDS }, () => 0n);
  const scalars = new Float32Array(SCALAR_DIM); // zeros
  const mask = new Uint8Array(NUM_ACTIONS);
  mask[0] = 1; // mark fold as legal so at least one action is allowed
  const h = new Float32Array(stateWidth);
  const c = new Float32Array(stateWidth);

  const feeds = {
    cards: new ort.Tensor("int64", cards, [1, NUM_CARDS]),
    scalars: new ort.Tensor("float32", scalars, [1, SCALAR_DIM]),
    action_mask: new ort.Tensor("bool", mask, [1, NUM_ACTIONS]),
    h: new ort.Tensor("float32", h, [1, 1, stateWidth]),
    c: new ort.Tensor("float32", c, [1, 1, stateWidth]),
  };

  const out = await session.run(feeds);

  // Output shapes / types we rely on in inference.mjs::act().
  assert.equal(out.action_logits.dims.join("x"), `1x${NUM_ACTIONS}`);
  assert.equal(out.value.data.length, 1, "value must be a single scalar per batch");
  assert.ok(Number.isFinite(out.value.data[0]), "value must be finite");
  for (let i = 0; i < NUM_ACTIONS; i++) {
    assert.ok(
      Number.isFinite(out.action_logits.data[i]),
      `action_logits[${i}] not finite`,
    );
  }
});
