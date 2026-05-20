// Unit tests for the heads-up NLHE engine. Validates core mechanics: blinds,
// position, call/check/raise/fold flow, stage transitions, all-in, showdown.
//
// Run: node web/tests/engine.test.mjs

import {
  newHand,
  applyAction,
  legalActions,
  isHandComplete,
  stateForEncoder,
  ACTION_FOLD,
  ACTION_CHECK,
  ACTION_CALL,
  ACTION_RAISE,
  STAGE_PREFLOP,
  STAGE_FLOP,
  STAGE_TURN,
  STAGE_RIVER,
  STAGE_TERMINAL,
} from "../src/engine.mjs";
import { stringToCard } from "../src/cards.mjs";

let pass = 0;
let fail = 0;
const failures = [];

function check(name, fn) {
  try {
    fn();
    pass++;
  } catch (e) {
    fail++;
    failures.push({ name, error: e });
    console.log(`FAIL: ${name}: ${e.message}`);
  }
}

function eq(actual, expected, msg = "") {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  if (a !== e) throw new Error(`${msg} expected ${e}, got ${a}`);
}

function truthy(v, msg = "") {
  if (!v) throw new Error(`${msg} expected truthy, got ${JSON.stringify(v)}`);
}

// ---- Initial deal ----

check("initial deal: blinds posted, SB to act", () => {
  const s = newHand({ button: 0, smallBlind: 5, bigBlind: 10, stacks: [1000, 1000] });
  eq(s.stage, STAGE_PREFLOP);
  eq(s.button, 0);
  eq(s.active_player, 0, "SB acts first preflop in HU");
  eq(s.bets, [5, 10]);
  eq(s.stacks, [995, 990]);
  eq(s.last_raise, 10, "initial min-raise is BB");
});

check("initial deal: BB on button rotates", () => {
  const s = newHand({ button: 1, smallBlind: 5, bigBlind: 10, stacks: [1000, 1000] });
  eq(s.button, 1);
  eq(s.active_player, 1, "P1 on button is SB and acts first");
  eq(s.bets, [10, 5]); // P0 is BB
  eq(s.stacks, [990, 995]);
});

check("legal actions: SB preflop facing BB", () => {
  const s = newHand({ button: 0, smallBlind: 5, bigBlind: 10 });
  const la = legalActions(s);
  eq(la.action_mask, [true, false, true, true], "FOLD, CALL, RAISE legal; no CHECK");
  eq(la.min_raise, 10);
  // Max raise = SB can put in (995 - to_call=5 = 990) on top of call
  eq(la.max_raise, 990);
});

// ---- Preflop fold ----

check("SB folds preflop -> BB wins blinds", () => {
  const s = newHand({ button: 0, smallBlind: 5, bigBlind: 10 });
  applyAction(s, ACTION_FOLD);
  eq(s.stage, STAGE_TERMINAL);
  eq(s.winner, 1, "BB wins");
  eq(s.stacks[1], 1005, "BB up 5");
  eq(s.stacks[0], 995, "SB down 5");
});

// ---- Preflop limp-check ----

check("SB limps, BB checks option -> flop dealt", () => {
  const s = newHand({ button: 0 });
  applyAction(s, ACTION_CALL); // SB calls (limp)
  eq(s.active_player, 1, "BB to act next");
  eq(s.stage, STAGE_PREFLOP, "still preflop until BB exercises option");
  const la = legalActions(s);
  eq(la.action_mask, [true, true, false, true], "BB can CHECK option");
  applyAction(s, ACTION_CHECK); // BB checks option
  eq(s.stage, STAGE_FLOP, "advanced to flop after BB checks option");
  eq(s.pot, 20, "pot = 2 * BB after limp");
  eq(s.bets, [0, 0]);
  eq(s.active_player, 1, "BB acts first postflop");
});

// ---- Preflop raise / call ----

check("SB raises, BB calls -> flop", () => {
  const s = newHand({ button: 0, smallBlind: 5, bigBlind: 10 });
  applyAction(s, ACTION_RAISE, 20); // delta 20 on top of 5-chip call -> +25 chips
  eq(s.bets, [30, 10], "SB invested 30 total (5 blind + 25 call/raise)");
  eq(s.active_player, 1);
  applyAction(s, ACTION_CALL);
  eq(s.stage, STAGE_FLOP);
  eq(s.pot, 60, "pot = 2 * 30");
});

// ---- Showdown ----

check("Showdown: P0 has AA, P1 has KK, board not connecting -> P0 wins", () => {
  // Construct deck so:
  //   hole order = [sb h0, bb h0, sb h1, bb h1, flop 0..2, turn, river]
  // We want P0 (SB) = AsAh, P1 (BB) = KsKh, board = 2c 3c 4c 5c 7d
  const deck = [
    stringToCard("As"), stringToCard("Ks"),
    stringToCard("Ah"), stringToCard("Kh"),
    stringToCard("2c"), stringToCard("3c"), stringToCard("4c"),
    stringToCard("5c"), stringToCard("7d"),
    // rest of deck doesn't matter for showdown
    ...Array.from({ length: 43 }, (_, i) => i + 13),
  ];
  const s = newHand({ button: 0, deck });
  // Both check/call to showdown
  applyAction(s, ACTION_CALL); // SB limps
  applyAction(s, ACTION_CHECK); // BB checks
  // Flop
  applyAction(s, ACTION_CHECK);
  applyAction(s, ACTION_CHECK);
  // Turn
  applyAction(s, ACTION_CHECK);
  applyAction(s, ACTION_CHECK);
  // River
  applyAction(s, ACTION_CHECK);
  applyAction(s, ACTION_CHECK);
  eq(s.stage, STAGE_TERMINAL);
  eq(s.winner, 0, "P0 with AA beats KK");
  eq(s.stacks[0], 1010);
  eq(s.stacks[1], 990);
});

// ---- All-in ----

check("All-in preflop runs out board to showdown", () => {
  const deck = [
    stringToCard("As"), stringToCard("Ks"),
    stringToCard("Ah"), stringToCard("Kh"),
    stringToCard("2c"), stringToCard("3c"), stringToCard("4c"),
    stringToCard("5c"), stringToCard("7d"),
    ...Array.from({ length: 43 }, (_, i) => i + 13),
  ];
  const s = newHand({ button: 0, deck, stacks: [100, 100], smallBlind: 5, bigBlind: 10 });
  // SB shoves all-in
  applyAction(s, ACTION_RAISE, 90); // 5 call + 90 = 95 (but stack is only 95 left, so all-in)
  eq(s.stacks[0], 0);
  eq(s.bets[0], 100);
  applyAction(s, ACTION_CALL);
  eq(s.stage, STAGE_TERMINAL, "skipped straight to showdown");
  eq(s.winner, 0);
});

// ---- Legal actions post-raise ----

check("After raise, opponent can fold/call/re-raise", () => {
  const s = newHand({ button: 0, smallBlind: 5, bigBlind: 10 });
  applyAction(s, ACTION_RAISE, 30); // SB raises to 35
  const la = legalActions(s);
  eq(la.action_mask, [true, false, true, true]);
  eq(la.min_raise, 30, "min re-raise = size of last raise");
});

// ---- stateForEncoder shape ----

check("stateForEncoder hides unrevealed community cards", () => {
  const s = newHand({ button: 0 });
  const enc = stateForEncoder(s);
  eq(enc.community_cards, [-1, -1, -1, -1, -1], "no board preflop");
});

check("stateForEncoder reveals flop after stage advance", () => {
  const deck = [
    stringToCard("As"), stringToCard("Ks"),
    stringToCard("Ah"), stringToCard("Kh"),
    stringToCard("2c"), stringToCard("3c"), stringToCard("4c"),
    stringToCard("5c"), stringToCard("7d"),
    ...Array.from({ length: 43 }, (_, i) => i + 13),
  ];
  const s = newHand({ button: 0, deck });
  applyAction(s, ACTION_CALL);
  applyAction(s, ACTION_CHECK);
  const enc = stateForEncoder(s);
  eq(enc.community_cards.slice(0, 3).map((c) => c >= 0), [true, true, true]);
  eq(enc.community_cards.slice(3), [-1, -1]);
});

// ---- Encoder integration ----

import { encodeObs } from "../src/encoder.mjs";

check("engine state flows through encoder without errors", () => {
  const s = newHand({ button: 0, smallBlind: 5, bigBlind: 10, startingStack: 1000 });
  const obs = encodeObs(stateForEncoder(s));
  eq(obs.cards.length, 7);
  eq(obs.scalars.length, 17);
  eq(obs.action_mask.length, 4);
  // SB position
  eq(obs.scalars[0], 0.0);
  // Hole cards visible, no board
  truthy(obs.cards[0] >= 0 && obs.cards[1] >= 0, "hole cards visible");
  eq(obs.cards.slice(2), [-1, -1, -1, -1, -1]);
  // SB facing BB: FOLD/CALL/RAISE legal, no CHECK
  eq(obs.action_mask, [true, false, true, true]);
});

check("engine state encodes correctly through full hand", () => {
  const s = newHand({ button: 0, smallBlind: 5, bigBlind: 10 });
  // Play a few actions and check obs at each step.
  for (const { action, amount } of [
    { action: ACTION_RAISE, amount: 20 }, // SB raises
    { action: ACTION_CALL }, // BB calls
    { action: ACTION_CHECK }, // BB acts first on flop
    { action: ACTION_RAISE, amount: 30 }, // SB raises flop
    { action: ACTION_FOLD }, // BB folds
  ]) {
    if (isHandComplete(s)) break;
    const obs = encodeObs(stateForEncoder(s));
    // All obs values must be finite and within expected ranges
    for (const sc of obs.scalars) {
      if (!Number.isFinite(sc)) throw new Error("non-finite scalar");
      if (sc < -1 || sc > 1.0001) throw new Error(`scalar out of range: ${sc}`);
    }
    applyAction(s, action, amount ?? 0);
  }
  eq(s.stage, STAGE_TERMINAL);
  eq(s.winner, 0, "SB wins after BB folds postflop");
});

// ---- Summary ----

console.log(`\n${pass} pass, ${fail} fail`);
if (fail > 0) {
  for (const f of failures) console.log(`  FAIL ${f.name}: ${f.error.message}`);
  process.exit(1);
}
