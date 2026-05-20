// Heads-up no-limit Texas hold'em engine.
//
// Produces a state shape compatible with web/src/encoder.mjs::encodeObs.
//
// Conventions (matching the Warp env):
//   Players are 0 and 1.
//   `button` is the player on the button (and small blind in heads-up).
//   Preflop: button (SB) acts first.
//   Postflop: non-button (BB) acts first.
//   Stages: 0=PREFLOP, 1=FLOP, 2=TURN, 3=RIVER, 4=SHOWDOWN, 5=TERMINAL.
//   Action codes: 0=FOLD, 1=CHECK, 2=CALL, 3=RAISE.

import pokersolver from "pokersolver";
const { Hand } = pokersolver;
import { cardToString, shuffledDeck, INVALID_CARD } from "./cards.mjs";

export const ACTION_FOLD = 0;
export const ACTION_CHECK = 1;
export const ACTION_CALL = 2;
export const ACTION_RAISE = 3;

export const STAGE_PREFLOP = 0;
export const STAGE_FLOP = 1;
export const STAGE_TURN = 2;
export const STAGE_RIVER = 3;
export const STAGE_SHOWDOWN = 4;
export const STAGE_TERMINAL = 5;

const STAGE_COMMUNITY_COUNT = { 0: 0, 1: 3, 2: 4, 3: 5, 4: 5, 5: 5 };

/** Create a fresh hand. button=0 means P0 is on the button (SB). */
export function newHand({
  button = 0,
  stacks = [1000, 1000],
  smallBlind = 5,
  bigBlind = 10,
  startingStack = 1000,
  rng = Math.random,
  deck = null, // optional: inject a pre-shuffled deck for tests
} = {}) {
  if (button !== 0 && button !== 1) throw new Error("button must be 0 or 1");
  const sb = button;
  const bb = 1 - button;

  const d = deck ?? shuffledDeck(rng);
  // Card order: hole_cards[sb][0], hole_cards[bb][0], hole_cards[sb][1],
  // hole_cards[bb][1], then flop[0..2], turn, river. (Matches real HU deal.)
  const holeCards = [
    [-1, -1],
    [-1, -1],
  ];
  holeCards[sb][0] = d[0];
  holeCards[bb][0] = d[1];
  holeCards[sb][1] = d[2];
  holeCards[bb][1] = d[3];
  const community = [d[4], d[5], d[6], d[7], d[8]];

  const initialStacks = [stacks[0] | 0, stacks[1] | 0];
  const newStacks = [initialStacks[0], initialStacks[1]];
  const bets = [0, 0];
  // Post blinds (capped at stack — short stacks post all-in).
  const sbPost = Math.min(smallBlind, newStacks[sb]);
  newStacks[sb] -= sbPost;
  bets[sb] = sbPost;
  const bbPost = Math.min(bigBlind, newStacks[bb]);
  newStacks[bb] -= bbPost;
  bets[bb] = bbPost;

  const state = {
    stage: STAGE_PREFLOP,
    button,
    active_player: sb, // SB acts first preflop in HU
    stacks: newStacks,
    bets,
    initial_stacks: initialStacks,
    pot: 0, // pot accumulates between streets; current street's chips live in `bets`
    hole_cards: holeCards,
    community_cards: community.slice(),
    visible_community: 0, // how many community cards are revealed
    last_raise: bigBlind, // initial min-raise is BB
    num_raises: 0,
    last_aggressor: -1,
    actions_this_street: 0,
    last_action_type: -1,
    last_action_amount: 0,
    last_action_was_raise: 0,
    cfg_starting_stack: startingStack,
    cfg_big_blind: bigBlind,
    cfg_small_blind: smallBlind,
    sb_post: sbPost,
    bb_post: bbPost,
    // Set when the hand ends:
    winner: null, // -1 (split) or player id, null while in progress
    rewards: [0, 0], // chip delta per player for this hand
  };

  // Mask the community cards based on stage (only flop visible at FLOP, etc.).
  applyVisibility(state);

  return state;
}

/**
 * Returns the slice of community cards that should be visible per
 * `state.visible_community`, plus -1 for hidden slots. This is the format the
 * obs encoder consumes (community_cards[i] = -1 if not yet dealt).
 */
function applyVisibility(state) {
  const visible = STAGE_COMMUNITY_COUNT[state.stage];
  state.visible_community = visible;
  // The actual community array stays full (we need it to deal further); the
  // state returned to the encoder will be derived via stateForEncoder().
}

/** Returns the state in the exact shape encodeObs() expects. */
export function stateForEncoder(state) {
  const visible = STAGE_COMMUNITY_COUNT[state.stage];
  const community = new Array(5);
  for (let i = 0; i < 5; i++) community[i] = i < visible ? state.community_cards[i] : -1;
  return {
    stage: state.stage,
    button: state.button,
    active_player: state.active_player,
    stacks: state.stacks.slice(),
    bets: state.bets.slice(),
    pot: state.pot,
    hole_cards: [state.hole_cards[0].slice(), state.hole_cards[1].slice()],
    community_cards: community,
    last_raise: state.last_raise,
    num_raises: state.num_raises,
    last_aggressor: state.last_aggressor,
    actions_this_street: state.actions_this_street,
    last_action_type: state.last_action_type,
    last_action_amount: state.last_action_amount,
    last_action_was_raise: state.last_action_was_raise,
    cfg_starting_stack: state.cfg_starting_stack,
    cfg_big_blind: state.cfg_big_blind,
  };
}

function toCall(state, player) {
  const opp = 1 - player;
  return Math.max(0, state.bets[opp] - state.bets[player]);
}

function minRaiseDelta(state) {
  let m = state.last_raise;
  let bb = state.cfg_big_blind;
  if (bb <= 0) bb = 10;
  if (m < bb) m = bb;
  return m;
}

function maxRaiseDelta(state, player) {
  const opp = 1 - player;
  const tc = toCall(state, player);
  const playerStack = state.stacks[player];
  const currentBet = state.bets[player];
  if (playerStack <= tc) return 0;
  const oppMaxTotal = state.bets[opp] + state.stacks[opp];
  const playerMaxTotal = currentBet + playerStack;
  const maxTotalAllowed = Math.min(oppMaxTotal, playerMaxTotal);
  return Math.max(0, maxTotalAllowed - (currentBet + tc));
}

/** Returns {action_mask: bool[4], min_raise: int, max_raise: int}. */
export function legalActions(state) {
  const player = state.active_player;
  const tc = toCall(state, player);
  const playerStack = state.stacks[player];

  const mask = [true, false, false, false]; // FOLD always
  mask[ACTION_CHECK] = tc === 0;
  mask[ACTION_CALL] = tc > 0 && playerStack > 0;

  const minD = minRaiseDelta(state);
  const maxD = maxRaiseDelta(state, player);
  const allInDelta = playerStack - tc;
  const existsRaise = maxD >= minD || (allInDelta > 0 && allInDelta <= maxD);
  mask[ACTION_RAISE] = existsRaise;

  return { action_mask: mask, min_raise: minD, max_raise: maxD };
}

/** True if both players have committed all chips. */
function bothAllIn(state) {
  return state.stacks[0] === 0 && state.stacks[1] === 0;
}

/** Whether the betting round is closed (action returns to the aggressor). */
function bettingRoundClosed(state) {
  // Both players must have acted at least once this street AND bets are equal.
  // - Preflop limp: SB calls (action 1, bets [BB,BB]), BB checks (action 2) -> closed.
  // - Preflop raise: SB raises (1, bets unequal), BB calls (2, bets equal) -> closed.
  // - Reraise: SB raises (1), BB raises (2, bets unequal), SB calls (3) -> closed.
  return state.actions_this_street >= 2 && state.bets[0] === state.bets[1];
}

function advanceStage(state) {
  if (state.stage === STAGE_PREFLOP) state.stage = STAGE_FLOP;
  else if (state.stage === STAGE_FLOP) state.stage = STAGE_TURN;
  else if (state.stage === STAGE_TURN) state.stage = STAGE_RIVER;
  else if (state.stage === STAGE_RIVER) state.stage = STAGE_SHOWDOWN;

  // Sweep current bets into pot.
  state.pot += state.bets[0] + state.bets[1];
  state.bets = [0, 0];
  // Reset street-local trackers.
  state.last_raise = 0;
  state.num_raises = 0;
  state.last_aggressor = -1;
  state.actions_this_street = 0;
  state.last_action_type = -1;
  state.last_action_amount = 0;
  state.last_action_was_raise = 0;

  // BB acts first postflop.
  if (state.stage !== STAGE_SHOWDOWN) {
    state.active_player = 1 - state.button;
    applyVisibility(state);
  }

  // If both players are all-in, skip straight to showdown by running
  // out the board.
  if (state.stage !== STAGE_SHOWDOWN && bothAllIn(state)) {
    while (state.stage !== STAGE_SHOWDOWN) {
      if (state.stage === STAGE_FLOP) state.stage = STAGE_TURN;
      else if (state.stage === STAGE_TURN) state.stage = STAGE_RIVER;
      else if (state.stage === STAGE_RIVER) state.stage = STAGE_SHOWDOWN;
    }
    applyVisibility(state);
  }

  if (state.stage === STAGE_SHOWDOWN) resolveShowdown(state);
}

function resolveShowdown(state) {
  const p0Cards = [...state.hole_cards[0], ...state.community_cards].map(cardToString);
  const p1Cards = [...state.hole_cards[1], ...state.community_cards].map(cardToString);
  const h0 = Hand.solve(p0Cards);
  const h1 = Hand.solve(p1Cards);
  const winners = Hand.winners([h0, h1]);
  let winner;
  if (winners.length === 2) winner = -1; // split
  else if (winners[0] === h0) winner = 0;
  else winner = 1;
  endHand(state, winner);
}

function endHand(state, winner) {
  state.winner = winner;
  const pot = state.pot + state.bets[0] + state.bets[1];
  if (winner === -1) {
    // Split pot — round down splits; remainder goes to BB by convention.
    const half = Math.floor(pot / 2);
    const rem = pot - 2 * half;
    state.stacks[0] += half;
    state.stacks[1] += half + rem; // BB gets odd chip
  } else {
    state.stacks[winner] += pot;
  }
  state.bets = [0, 0];
  state.pot = 0;
  state.stage = STAGE_TERMINAL;
  for (let p = 0; p < 2; p++) {
    state.rewards[p] = state.stacks[p] - state.initial_stacks[p];
  }
}

/**
 * Apply an action to the state. Mutates and returns the state.
 * `amount` is the raise *delta* (chips added on top of the call). Required
 * when action === ACTION_RAISE.
 */
export function applyAction(state, action, amount = 0) {
  if (state.stage === STAGE_TERMINAL || state.stage === STAGE_SHOWDOWN) {
    throw new Error("hand is over");
  }
  const player = state.active_player;
  const opp = 1 - player;
  const legal = legalActions(state);
  if (!legal.action_mask[action]) {
    throw new Error(`illegal action ${action} (mask=${JSON.stringify(legal.action_mask)})`);
  }

  if (action === ACTION_FOLD) {
    endHand(state, opp);
    return state;
  }

  if (action === ACTION_CHECK) {
    state.last_action_type = ACTION_CHECK;
    state.last_action_amount = 0;
    state.last_action_was_raise = 0;
    state.actions_this_street += 1;
  } else if (action === ACTION_CALL) {
    const tc = toCall(state, player);
    const paid = Math.min(tc, state.stacks[player]);
    state.stacks[player] -= paid;
    state.bets[player] += paid;
    state.last_action_type = ACTION_CALL;
    state.last_action_amount = paid;
    state.last_action_was_raise = 0;
    state.actions_this_street += 1;
  } else if (action === ACTION_RAISE) {
    // `amount` is the delta on top of the call. Clamp to legal range.
    let delta = amount | 0;
    const tc = toCall(state, player);
    const allInDelta = state.stacks[player] - tc;
    delta = Math.max(legal.min_raise, Math.min(delta, legal.max_raise));
    // All-in for less than min raise is allowed (treat as all-in).
    if (allInDelta > 0 && allInDelta < legal.min_raise) {
      delta = allInDelta;
    }
    const paid = tc + delta;
    const actualPaid = Math.min(paid, state.stacks[player]);
    state.stacks[player] -= actualPaid;
    state.bets[player] += actualPaid;

    // last_raise is the delta amount (used by min-raise rule).
    if (delta >= legal.min_raise) {
      state.last_raise = delta;
      state.last_aggressor = player;
      state.num_raises += 1;
    }
    state.last_action_type = ACTION_RAISE;
    state.last_action_amount = actualPaid;
    state.last_action_was_raise = 1;
    state.actions_this_street += 1;
  }

  // Pass turn to opp.
  state.active_player = opp;

  // Did the action close the round?
  if (bettingRoundClosed(state)) {
    advanceStage(state);
  }

  return state;
}

/** True if the hand has finished (either showdown resolved or fold). */
export function isHandComplete(state) {
  return state.stage === STAGE_TERMINAL;
}

/** Convenience: roll over the button for the next hand. */
export function nextButton(button) {
  return 1 - button;
}
