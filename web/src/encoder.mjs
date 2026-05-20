// Pure-function obs encoder mirroring the Warp kernels in:
//   src/gpu_poker/kernels/observations.py::write_observation
//   src/gpu_poker/kernels/legal_actions.py::write_legal_actions
//
// The output of `encodeObs(state)` must match the Warp env's outputs bit-for-bit
// (within float32 round-tripping) for the trained policy to play correctly in
// the browser. This is enforced by tests/encoder.test.mjs against fixtures
// dumped from the real Warp env (train/dump_obs_fixtures.py).
//
// State shape (one env's worth):
//   {
//     stage, button, active_player,
//     stacks: [int, int], bets: [int, int], pot,
//     hole_cards: [[int, int], [int, int]],  // [player][card]
//     community_cards: [int, int, int, int, int],
//     last_raise, num_raises, last_aggressor,
//     actions_this_street, last_action_type, last_action_amount, last_action_was_raise,
//     cfg_starting_stack, cfg_big_blind,
//   }
//
// Constants below mirror src/gpu_poker/constants.py.

export const ACTION_FOLD = 0;
export const ACTION_CHECK = 1;
export const ACTION_CALL = 2;
export const ACTION_RAISE = 3;
export const NUM_ACTIONS = 4;

export const STAGE_PREFLOP = 0;
export const STAGE_FLOP = 1;
export const STAGE_TURN = 2;
export const STAGE_RIVER = 3;

export const INVALID_CARD = -1;
const DEFAULT_STARTING_STACK = 1000;
const DEFAULT_BIG_BLIND = 10;

function getToCall(state, player) {
  // actions.get_to_call: max(0, opp.bet - player.bet)
  const opp = 1 - player;
  const diff = state.bets[opp] - state.bets[player];
  return diff < 0 ? 0 : diff;
}

function getVisibleCard(state, slot) {
  // Mirrors observations.get_visible_card: active player's hole cards (slots 0,1)
  // + community cards revealed by stage (slots 2-6).
  const player = state.active_player;
  if (slot < 2) {
    return state.hole_cards[player][slot];
  }
  const boardIdx = slot - 2;
  const stage = state.stage;
  if (boardIdx < 3) {
    return stage >= STAGE_FLOP ? state.community_cards[boardIdx] : INVALID_CARD;
  }
  if (boardIdx === 3) {
    return stage >= STAGE_TURN ? state.community_cards[boardIdx] : INVALID_CARD;
  }
  if (boardIdx === 4) {
    return stage >= STAGE_RIVER ? state.community_cards[boardIdx] : INVALID_CARD;
  }
  return INVALID_CARD;
}

// Warp uses fp32 internally; doing the arithmetic in fp64 then rounding to f32
// matches the recorded values within 1 ULP and avoids accumulated drift.
function toF32(x) {
  return Math.fround(x);
}

function encodeScalars(state) {
  const out = new Array(17).fill(0);
  const player = state.active_player;
  const opp = 1 - player;

  let startingStack = state.cfg_starting_stack;
  if (startingStack <= 0) startingStack = DEFAULT_STARTING_STACK;
  let bigBlind = state.cfg_big_blind;
  if (bigBlind <= 0) bigBlind = DEFAULT_BIG_BLIND;

  // 0: position (0 = Button/SB, 1 = BB)
  out[0] = player === state.button ? 0.0 : 1.0;

  // 1-2: stack / starting_stack
  out[1] = toF32(state.stacks[player] / startingStack);
  out[2] = toF32(state.stacks[opp] / startingStack);

  // 3-4: bets / pot_total
  const potTotal = state.pot + state.bets[0] + state.bets[1];
  if (potTotal > 0) {
    out[3] = toF32(state.bets[player] / potTotal);
    out[4] = toF32(state.bets[opp] / potTotal);
  } else {
    out[3] = 0.0;
    out[4] = 0.0;
  }

  // 5: pot_total / (2 * starting_stack)
  out[5] = toF32(potTotal / (startingStack * 2));

  // 6: effective_to_call / player_stack, clipped to [0, 1]
  const toCall = getToCall(state, player);
  const playerStack = state.stacks[player];
  if (playerStack > 0) {
    const effectiveToCall = Math.min(toCall, playerStack);
    out[6] = toF32(effectiveToCall / playerStack);
  } else {
    out[6] = 0.0;
  }

  // 7: stage / 5.0
  out[7] = toF32(state.stage / 5.0);

  // 8: last_raise / starting_stack
  out[8] = toF32(state.last_raise / startingStack);

  // 9: num_raises / 4.0
  out[9] = toF32(state.num_raises / 4.0);

  // 10: last_aggressor indicator
  if (state.last_aggressor === player) out[10] = 1.0;
  else if (state.last_aggressor === opp) out[10] = 0.5;
  else out[10] = 0.0;

  // 11: actions_this_street (clamped to [0, 15]) / 15.0
  let actsStreet = state.actions_this_street;
  if (actsStreet < 0) actsStreet = 0;
  if (actsStreet > 15) actsStreet = 15;
  out[11] = toF32(actsStreet / 15.0);

  // 12: last_action_type: 0.0 if -1, else (type + 1) / 4.0
  const lat = state.last_action_type;
  out[12] = lat < 0 ? 0.0 : toF32((lat + 1) / 4.0);

  // 13: last_action_was_raise (0/1)
  out[13] = toF32(state.last_action_was_raise);

  // 14: last_action_amount / starting_stack, clipped to [0, 1]
  let amt = state.last_action_amount;
  if (amt < 0) amt = 0;
  let amtF = toF32(amt / startingStack);
  if (amtF > 1.0) amtF = 1.0;
  out[14] = amtF;

  // 15: big_blind / starting_stack
  out[15] = toF32(bigBlind / startingStack);

  // 16: effective stack in BB / 200, clipped to [0, 1]
  let eff = state.stacks[0];
  if (state.stacks[1] < eff) eff = state.stacks[1];
  const effBb = bigBlind > 0 ? eff / bigBlind : 0.0;
  let effBbNorm = toF32(effBb / 200.0);
  if (effBbNorm > 1.0) effBbNorm = 1.0;
  out[16] = effBbNorm;

  return out;
}

function minRaiseDelta(state) {
  let m = state.last_raise;
  let bb = state.cfg_big_blind;
  if (bb <= 0) bb = DEFAULT_BIG_BLIND;
  if (m < bb) m = bb;
  return m;
}

function maxRaiseDelta(state) {
  const player = state.active_player;
  const opp = 1 - player;
  const toCall = getToCall(state, player);
  const playerStack = state.stacks[player];
  const currentBet = state.bets[player];

  if (playerStack <= toCall) return 0;

  const oppMaxTotal = state.bets[opp] + state.stacks[opp];
  const playerMaxTotal = currentBet + playerStack;

  let maxTotalAllowed = oppMaxTotal;
  if (playerMaxTotal < maxTotalAllowed) maxTotalAllowed = playerMaxTotal;

  let maxDelta = maxTotalAllowed - (currentBet + toCall);
  if (maxDelta < 0) maxDelta = 0;
  return maxDelta;
}

function encodeLegalActions(state) {
  const player = state.active_player;
  const toCall = getToCall(state, player);
  const playerStack = state.stacks[player];

  const mask = new Array(NUM_ACTIONS).fill(false);
  mask[ACTION_FOLD] = true;
  mask[ACTION_CHECK] = toCall === 0;
  mask[ACTION_CALL] = toCall > 0 && playerStack > 0;

  const minD = minRaiseDelta(state);
  const maxD = maxRaiseDelta(state);
  const allInDelta = playerStack - toCall;
  const existsRaise = maxD >= minD || (allInDelta > 0 && allInDelta <= maxD);
  mask[ACTION_RAISE] = existsRaise;

  return { action_mask: mask, min_raise: minD, max_raise: maxD };
}

export function encodeCards(state) {
  const out = new Array(7);
  for (let i = 0; i < 7; i++) out[i] = getVisibleCard(state, i);
  return out;
}

export function encodeObs(state) {
  const { action_mask, min_raise, max_raise } = encodeLegalActions(state);
  return {
    cards: encodeCards(state),
    scalars: encodeScalars(state),
    action_mask,
    min_raise,
    max_raise,
    // Plain mirrors of fields needed by the transformer's JS-side
    // pushActionToken(); harmless for the LSTM path.
    player_id: state.active_player,
    stage: state.stage,
  };
}
