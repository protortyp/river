// Parity test: load fixtures generated from the real Warp env and assert that
// our JS encoder produces the same outputs. If this fails, the trained policy
// will misread the game state in the browser.
//
// Run: node web/tests/encoder.test.mjs

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { encodeObs } from "../src/encoder.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const FIXTURE_PATH = resolve(__dirname, "fixtures.json");

const payload = JSON.parse(readFileSync(FIXTURE_PATH, "utf8"));
const fixtures = payload.fixtures;
console.log(`[test] loaded ${fixtures.length} fixtures from ${FIXTURE_PATH}`);

const SCALAR_EPS = 1e-6;

let failures = 0;
const maxDiffs = {
  cards: 0,
  scalars: 0,
  action_mask: 0,
  min_raise: 0,
  max_raise: 0,
};
const failureExamples = [];

for (let i = 0; i < fixtures.length; i++) {
  const { state, obs: expected } = fixtures[i];
  const got = encodeObs(state);

  const errs = [];

  // Cards: exact int match (-1 for invisible).
  for (let j = 0; j < 7; j++) {
    if (got.cards[j] !== expected.cards[j]) {
      const d = Math.abs(got.cards[j] - expected.cards[j]);
      maxDiffs.cards = Math.max(maxDiffs.cards, d);
      errs.push(`cards[${j}]: got=${got.cards[j]} expected=${expected.cards[j]}`);
    }
  }

  // Scalars: float compare with small epsilon (fixtures rounded to 9 decimals,
  // both sides go through fp32 rounding).
  for (let j = 0; j < 17; j++) {
    const d = Math.abs(got.scalars[j] - expected.scalars[j]);
    maxDiffs.scalars = Math.max(maxDiffs.scalars, d);
    if (d > SCALAR_EPS) {
      errs.push(`scalars[${j}]: got=${got.scalars[j]} expected=${expected.scalars[j]} diff=${d.toExponential(2)}`);
    }
  }

  // Action mask: exact bool match.
  for (let j = 0; j < 4; j++) {
    if (got.action_mask[j] !== expected.action_mask[j]) {
      maxDiffs.action_mask = 1;
      errs.push(`action_mask[${j}]: got=${got.action_mask[j]} expected=${expected.action_mask[j]}`);
    }
  }

  // min/max raise: exact int match.
  if (got.min_raise !== expected.min_raise) {
    const d = Math.abs(got.min_raise - expected.min_raise);
    maxDiffs.min_raise = Math.max(maxDiffs.min_raise, d);
    errs.push(`min_raise: got=${got.min_raise} expected=${expected.min_raise}`);
  }
  if (got.max_raise !== expected.max_raise) {
    const d = Math.abs(got.max_raise - expected.max_raise);
    maxDiffs.max_raise = Math.max(maxDiffs.max_raise, d);
    errs.push(`max_raise: got=${got.max_raise} expected=${expected.max_raise}`);
  }

  if (errs.length > 0) {
    failures++;
    if (failureExamples.length < 3) {
      failureExamples.push({ index: i, state, expected, got, errs });
    }
  }
}

console.log(`[test] max abs diffs across ${fixtures.length} fixtures:`);
console.log(`  cards         : ${maxDiffs.cards}`);
console.log(`  scalars       : ${maxDiffs.scalars.toExponential(3)}`);
console.log(`  action_mask   : ${maxDiffs.action_mask}`);
console.log(`  min_raise     : ${maxDiffs.min_raise}`);
console.log(`  max_raise     : ${maxDiffs.max_raise}`);

if (failures > 0) {
  console.log(`\n[test] FAIL: ${failures}/${fixtures.length} fixtures mismatched`);
  for (const ex of failureExamples) {
    console.log(`\n--- fixture ${ex.index} ---`);
    console.log(`state:`, JSON.stringify(ex.state, null, 2));
    console.log(`errors:`);
    for (const e of ex.errs) console.log(`  ${e}`);
  }
  process.exit(1);
}

console.log(`[test] PASS (${fixtures.length} fixtures)`);
