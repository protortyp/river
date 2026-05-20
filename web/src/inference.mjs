// onnxruntime-web inference wrapper + action sampling from model outputs.
//
// The exported ONNX model returns:
//   action_logits: [B, 4]   — already masked (illegal actions = -inf)
//   raise_logits:  [B, NUM_RAISE_BUCKETS] — categorical logits over discrete
//                  pot-fraction buckets (matches src/gpu_poker/policy.py).
//   value:         [B]
//   h_new, c_new:  [1, B, H]    (LSTM: new hidden state)
//                                (transformer: passthrough -- pushActionToken
//                                 must run in JS after sampling to update
//                                 the per-env token table for the next call)
//
// Sampling happens in JS so we can choose temperature / deterministic mode
// without re-exporting the model. The caller is responsible for mapping the
// chosen bucket index to a chip amount via the engine's pot/min/max raise
// values (see RAISE_BUCKET_FRACTIONS below).
//
// Backbone detection: the loaded ONNX file carries `policy_backbone` and
// `state_width` keys in model.metadata_props (set by train/export_onnx.py).
// Falls back to the legacy LSTM default (state_width=384) when metadata is
// missing, so old pre-transformer exports keep working unchanged.

import * as ort from "onnxruntime-web";
import { encodeObs } from "./encoder.mjs";

// Legacy LSTM default when ONNX metadata is missing. New transformer exports
// override via model.metadata_props.state_width = 129 (STATE_WIDTH from
// src/gpu_poker/policy_transformer.py).
const LSTM_HIDDEN_DEFAULT = 384;
const NUM_CARDS = 7;
const SCALAR_DIM = 17;
const NUM_ACTIONS = 4;

// --- Transformer packed-state constants (mirror src/gpu_poker/policy_transformer.py) ---
// State layout: [tokens.flatten() (MAX_TOKENS * 4), length] = STATE_WIDTH
// Stored as fp32 because the buffer is fp32, but values are tiny ints so
// round-trip is exact.
const TRANSFORMER_MAX_TOKENS = 32;
const TRANSFORMER_TOKEN_FIELDS = 4;
const TRANSFORMER_STATE_WIDTH =
  TRANSFORMER_MAX_TOKENS * TRANSFORMER_TOKEN_FIELDS + 1;
// 1-indexed vocab caps (PAD=0). Must match policy_transformer constants.
const TRANSFORMER_PLAYER_VOCAB = 3; // PAD + p0 + p1
const TRANSFORMER_ACTION_VOCAB = NUM_ACTIONS + 1;
const TRANSFORMER_BUCKET_VOCAB = 7 + 1; // NUM_RAISE_BUCKETS + 1
const TRANSFORMER_STAGE_VOCAB = 4 + 1; // NUM_STAGES (PRE/FLOP/TURN/RIVER) + 1

// Discrete raise-size buckets. Must match src/gpu_poker/policy.py exactly.
// Sentinels (negative values) mean "use min-raise" / "use max-raise (all-in)".
export const MIN_RAISE_BUCKET = -1.0;
export const ALL_IN_BUCKET = -2.0;
export const RAISE_BUCKET_FRACTIONS = [
  MIN_RAISE_BUCKET,  // bucket 0: min-raise
  0.5,               // bucket 1: 0.5 * pot
  0.75,              // bucket 2: 0.75 * pot
  1.0,               // bucket 3: 1.0 * pot
  1.5,               // bucket 4: 1.5 * pot
  2.5,               // bucket 5: 2.5 * pot
  ALL_IN_BUCKET,     // bucket 6: all-in
];
export const NUM_RAISE_BUCKETS = RAISE_BUCKET_FRACTIONS.length;

/** Map a discrete bucket to a chip-delta raise amount, clamped to [min, max]. */
export function bucketToAmount({ bucket, potTotal, minRaise, maxRaise }) {
  const frac = RAISE_BUCKET_FRACTIONS[bucket];
  let amt;
  if (frac === MIN_RAISE_BUCKET) amt = minRaise;
  else if (frac === ALL_IN_BUCKET) amt = maxRaise;
  else amt = Math.round(frac * potTotal);
  // All-in only situations: max_raise < min_raise; force the all-in delta.
  if (maxRaise < minRaise) amt = maxRaise;
  const lo = Math.min(minRaise, maxRaise);
  const hi = Math.max(minRaise, maxRaise);
  return Math.max(lo, Math.min(hi, amt));
}

export class PokerBot {
  constructor() {
    this.session = null;
    this.h = null; // LSTM hidden state OR transformer packed state, per-hand
    this.c = null;
    // Per-model: filled at loadModel() time from ONNX metadata.
    this.backbone = "lstm";
    this.stateWidth = LSTM_HIDDEN_DEFAULT;
    // Backwards-compat alias kept so older callers (engine.mjs, smoke.mjs)
    // that reference bot.lstmHidden don't break.
    this.lstmHidden = LSTM_HIDDEN_DEFAULT;
  }

  /** Load a model from a File (drag-and-drop), URL, or ArrayBuffer. */
  async loadModel(source) {
    let bytes;
    if (source instanceof File || source instanceof Blob) {
      bytes = new Uint8Array(await source.arrayBuffer());
    } else if (source instanceof ArrayBuffer) {
      bytes = new Uint8Array(source);
    } else if (typeof source === "string") {
      const r = await fetch(source);
      bytes = new Uint8Array(await r.arrayBuffer());
    } else if (source instanceof Uint8Array) {
      bytes = source;
    } else {
      throw new Error(`unsupported model source type: ${typeof source}`);
    }

    // Configure ORT-Web. Single-thread WASM (no SharedArrayBuffer needed -> no COOP/COEP headers).
    ort.env.wasm.numThreads = 1;

    this.session = await ort.InferenceSession.create(bytes, {
      executionProviders: ["wasm"],
    });

    // Sanity-check inputs and pick up backbone + state width from metadata.
    const expected = ["cards", "scalars", "action_mask", "h", "c"];
    for (const name of expected) {
      if (!this.session.inputNames.includes(name)) {
        throw new Error(`model is missing required input '${name}'`);
      }
    }

    // Detect backbone + state width from the `h` input's declared shape.
    // The exporter writes the right shape: LSTM -> [1, B, lstm_hidden],
    // transformer -> [1, B, STATE_WIDTH=129]. We previously read
    // session.modelMetadata.customMetadataMap, but onnxruntime-web 1.20+
    // doesn't expose modelMetadata at all (the property is undefined),
    // so that code silently fell back to LSTM and passed wrong-shaped
    // tensors at run() time. Reading the input shape is robust across
    // versions.
    const hMeta = this.session.inputMetadata?.h ?? this.session.inputMetadata?.[3];
    const hShape = hMeta && Array.isArray(hMeta.shape) ? hMeta.shape : null;
    const hWidth =
      hShape && typeof hShape[2] === "number" && hShape[2] > 0 ? hShape[2] : null;
    if (hWidth === TRANSFORMER_STATE_WIDTH) {
      this.backbone = "transformer";
      this.stateWidth = TRANSFORMER_STATE_WIDTH;
    } else if (hWidth && hWidth !== TRANSFORMER_STATE_WIDTH) {
      this.backbone = "lstm";
      this.stateWidth = hWidth;
    } else {
      // Last-resort fallback: legacy LSTM at the historical hidden size.
      this.backbone = "lstm";
      this.stateWidth = LSTM_HIDDEN_DEFAULT;
    }
    this.lstmHidden = this.stateWidth; // backwards-compat alias

    this.resetHidden();
  }

  /** Reset state at the start of every hand. Works for both backbones --
   *  LSTM zero hidden state, or transformer empty packed state (PAD tokens
   *  + length=0). */
  resetHidden() {
    this.h = new Float32Array(1 * 1 * this.stateWidth);
    this.c = new Float32Array(1 * 1 * this.stateWidth);
  }

  /**
   * Run inference on a single game state. Returns {actionType, raiseDelta, value, probs}.
   *
   * @param state Engine state (will be passed through encodeObs)
   * @param opts.deterministic If true, take argmax instead of sampling.
   * @param opts.temperature Softmax temperature (default 1.0).
   */
  async act(state, { deterministic = false, temperature = 1.0 } = {}) {
    if (!this.session) throw new Error("model not loaded");

    const obs = encodeObs(state);

    const cardsArr = new BigInt64Array(NUM_CARDS);
    for (let i = 0; i < NUM_CARDS; i++) cardsArr[i] = BigInt(obs.cards[i]);
    const scalarsArr = new Float32Array(obs.scalars);
    const maskArr = new Uint8Array(NUM_ACTIONS);
    for (let i = 0; i < NUM_ACTIONS; i++) maskArr[i] = obs.action_mask[i] ? 1 : 0;

    const feeds = {
      cards: new ort.Tensor("int64", cardsArr, [1, NUM_CARDS]),
      scalars: new ort.Tensor("float32", scalarsArr, [1, SCALAR_DIM]),
      action_mask: new ort.Tensor("bool", maskArr, [1, NUM_ACTIONS]),
      h: new ort.Tensor("float32", this.h, [1, 1, this.stateWidth]),
      c: new ort.Tensor("float32", this.c, [1, 1, this.stateWidth]),
    };

    const out = await this.session.run(feeds);

    // Update per-hand state.
    //   LSTM:        h_new/c_new are the post-LSTM state directly.
    //   transformer: h_new is a passthrough; we must push the chosen action
    //                token onto the per-env table AFTER sampling below.
    this.h = new Float32Array(out.h_new.data);
    this.c = new Float32Array(out.c_new.data);

    const logits = Array.from(out.action_logits.data);
    const raiseLogits = Array.from(out.raise_logits.data);
    const value = out.value.data[0];

    const probs = softmaxStable(logits.map((l) => l / temperature));
    const actionType = deterministic ? argmax(logits) : sampleCategorical(probs);

    const raiseProbs = softmaxStable(raiseLogits.map((l) => l / temperature));
    const raiseBucket = deterministic ? argmax(raiseLogits) : sampleCategorical(raiseProbs);

    if (this.backbone === "transformer") {
      // Mirror push_action_token_packed: record the action we just chose
      // into the per-env token buffer so it shows up in the history for
      // the next call. stage comes from the encoded observation (the env
      // exposes it as the last few scalar fields in obs.scalars; see
      // src/gpu_poker/policy_transformer.py:_stage_from_scalars for the
      // server-side mirror).
      const playerId = obs.player_id ?? 0;
      const stage = obs.stage ?? 0;
      this.h = pushActionToken(this.h, {
        playerId,
        actionType,
        raiseBucket,
        stage,
      });
    }

    return {
      actionType,
      raiseBucket,
      value,
      probs,
      raiseProbs,
    };
  }
}

/**
 * In-JS mirror of src/gpu_poker/policy_transformer.py::push_action_token_packed
 * for a SINGLE env (batch=1). Append one public-action token to the per-env
 * packed state buffer.
 *
 * @param {Float32Array} h  Length STATE_WIDTH packed state buffer (will be cloned).
 * @param {object} opts     {playerId, actionType, raiseBucket, stage} -- all small ints.
 * @returns {Float32Array}  New packed state with the token appended + length++.
 */
export function pushActionToken(h, { playerId, actionType, raiseBucket, stage }) {
  if (h.length !== TRANSFORMER_STATE_WIDTH) {
    throw new Error(
      `pushActionToken: expected state length ${TRANSFORMER_STATE_WIDTH}, got ${h.length}`,
    );
  }
  const out = new Float32Array(h);
  // Length field lives at the last slot; round to int defensively.
  const lengthIdx = TRANSFORMER_STATE_WIDTH - 1;
  let length = Math.round(out[lengthIdx]);
  if (length >= TRANSFORMER_MAX_TOKENS) return out; // history full; silent drop (matches Python)
  // 1-indexed (PAD = 0). Match _embed_history's clamp ranges.
  const pid = Math.min(Math.max(playerId + 1, 0), TRANSFORMER_PLAYER_VOCAB - 1);
  const act = Math.min(Math.max(actionType + 1, 0), TRANSFORMER_ACTION_VOCAB - 1);
  const bkt = Math.min(Math.max(raiseBucket + 1, 0), TRANSFORMER_BUCKET_VOCAB - 1);
  const stg = Math.min(Math.max(stage + 1, 0), TRANSFORMER_STAGE_VOCAB - 1);
  const writeBase = length * TRANSFORMER_TOKEN_FIELDS;
  out[writeBase + 0] = pid;
  out[writeBase + 1] = act;
  out[writeBase + 2] = bkt;
  out[writeBase + 3] = stg;
  out[lengthIdx] = length + 1;
  return out;
}

// ---------- helpers ----------

function argmax(arr) {
  let best = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
  return best;
}

function softmaxStable(logits) {
  let m = -Infinity;
  for (const l of logits) if (l > m) m = l;
  let sum = 0;
  const out = new Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] - m);
    out[i] = e;
    sum += e;
  }
  if (sum === 0) {
    // All -inf (no legal action) — fall back to uniform.
    return out.map(() => 1 / logits.length);
  }
  for (let i = 0; i < logits.length; i++) out[i] /= sum;
  return out;
}

function sampleCategorical(probs) {
  const r = Math.random();
  let cum = 0;
  for (let i = 0; i < probs.length; i++) {
    cum += probs[i];
    if (r < cum) return i;
  }
  return probs.length - 1;
}

