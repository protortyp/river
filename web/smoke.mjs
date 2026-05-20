// Smoke test: load the exported .onnx in onnxruntime-web (the same runtime
// the browser will use) and run a single forward pass. If this passes, we
// can confidently move on to building the UI.

import * as ort from "onnxruntime-web";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const MODEL_PATH = process.argv[2] || "/tmp/test_model.onnx";

const LSTM_HIDDEN = 384;
const SCALAR_DIM = 17;
const NUM_CARDS = 7;
const NUM_ACTIONS = 4;
const BATCH = 3;

console.log(`[smoke] loading ${MODEL_PATH}`);
const modelBytes = readFileSync(MODEL_PATH);
console.log(`[smoke] model size: ${(modelBytes.byteLength / 1024 / 1024).toFixed(2)} MB`);

// Force WASM backend (what the browser will use too).
ort.env.wasm.numThreads = 1;

const t0 = performance.now();
const session = await ort.InferenceSession.create(modelBytes, {
  executionProviders: ["wasm"],
});
console.log(`[smoke] session loaded in ${(performance.now() - t0).toFixed(1)}ms`);
console.log(`[smoke] inputs: ${session.inputNames.join(", ")}`);
console.log(`[smoke] outputs: ${session.outputNames.join(", ")}`);

// Build dummy inputs (deterministic so we can spot-check outputs).
function makeInputs(batch) {
  // cards: random in [-1, 51], i64
  const cards = new BigInt64Array(batch * NUM_CARDS);
  for (let i = 0; i < cards.length; i++) cards[i] = BigInt(((i * 7) % 53) - 1);

  // scalars: float32, small values
  const scalars = new Float32Array(batch * SCALAR_DIM);
  for (let i = 0; i < scalars.length; i++) scalars[i] = ((i % 17) - 8) * 0.1;

  // action_mask: at least FOLD legal + one random extra. ORT-Web bool is Uint8.
  const mask = new Uint8Array(batch * NUM_ACTIONS);
  for (let b = 0; b < batch; b++) {
    mask[b * NUM_ACTIONS + 0] = 1; // FOLD
    mask[b * NUM_ACTIONS + ((b + 1) % NUM_ACTIONS)] = 1;
  }

  const hData = new Float32Array(1 * batch * LSTM_HIDDEN); // zeros
  const cData = new Float32Array(1 * batch * LSTM_HIDDEN); // zeros

  return {
    cards: new ort.Tensor("int64", cards, [batch, NUM_CARDS]),
    scalars: new ort.Tensor("float32", scalars, [batch, SCALAR_DIM]),
    action_mask: new ort.Tensor("bool", mask, [batch, NUM_ACTIONS]),
    h: new ort.Tensor("float32", hData, [1, batch, LSTM_HIDDEN]),
    c: new ort.Tensor("float32", cData, [1, batch, LSTM_HIDDEN]),
  };
}

const feeds = makeInputs(BATCH);

// Warmup (first run does some lazy work).
await session.run(feeds);

// Timed runs.
const N = 50;
const start = performance.now();
let out;
for (let i = 0; i < N; i++) out = await session.run(feeds);
const elapsed = performance.now() - start;
console.log(`[smoke] ${N} runs in ${elapsed.toFixed(1)}ms  =>  ${(elapsed / N).toFixed(2)}ms/run avg (batch=${BATCH})`);

// Sanity check outputs.
console.log(`[smoke] action_logits shape: [${out.action_logits.dims.join(", ")}]`);
console.log(`[smoke] raise_logits shape:  [${out.raise_logits.dims.join(", ")}]`);
console.log(`[smoke] value shape:         [${out.value.dims.join(", ")}]`);
console.log(`[smoke] h_new shape:         [${out.h_new.dims.join(", ")}]`);
console.log(`[smoke] c_new shape:         [${out.c_new.dims.join(", ")}]`);

// Spot check: the masked actions should be -inf (or near it).
const logits = out.action_logits.data;
console.log(`[smoke] sample action_logits row 0: [${Array.from(logits.slice(0, 4)).map(x => x.toFixed(3)).join(", ")}]`);
console.log(`[smoke] sample value row 0..${BATCH - 1}: [${Array.from(out.value.data).map(x => x.toFixed(4)).join(", ")}]`);

console.log("[smoke] PASS");
