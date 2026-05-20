import warp as wp

from gpu_poker import constants as c
from gpu_poker.kernels import actions
from gpu_poker.struct_types import GameState


@wp.func
def _min_raise_delta(state: GameState, env_idx: wp.int32) -> wp.int32:
    m = state.last_raise[env_idx]
    bb = state.cfg_big_blind[env_idx]
    if bb <= 0:
        bb = c.BIG_BLIND
    if m < bb:
        m = bb
    return m


@wp.func
def _max_raise_delta(state: GameState, env_idx: wp.int32, player: wp.int32) -> wp.int32:
    opp = 1 - player
    to_call = actions.get_to_call(state, env_idx, player)
    player_stack = state.stacks[env_idx, player]
    current_bet = state.bets[env_idx, player]

    if player_stack <= to_call:
        return 0

    opp_max_total = state.bets[env_idx, opp] + state.stacks[env_idx, opp]
    player_max_total = current_bet + player_stack

    max_total_allowed = opp_max_total
    if player_max_total < max_total_allowed:
        max_total_allowed = player_max_total

    max_delta = max_total_allowed - (current_bet + to_call)
    if max_delta < 0:
        max_delta = 0
    return max_delta


@wp.func
def write_legal_actions(
    state: GameState,
    env_idx: wp.int32,
    action_mask: wp.array(dtype=wp.bool, ndim=2),  # [N_ENVS, 4]
    min_raise: wp.array(dtype=wp.int32),  # [N_ENVS] raise delta
    max_raise: wp.array(dtype=wp.int32),  # [N_ENVS] raise delta
):
    player = state.active_player[env_idx]
    to_call = actions.get_to_call(state, env_idx, player)
    player_stack = state.stacks[env_idx, player]

    action_mask[env_idx, c.ACTION_FOLD] = True
    action_mask[env_idx, c.ACTION_CHECK] = to_call == 0
    action_mask[env_idx, c.ACTION_CALL] = (to_call > 0) and (player_stack > 0)

    min_d = _min_raise_delta(state, env_idx)
    max_d = _max_raise_delta(state, env_idx, player)

    min_raise[env_idx] = min_d
    max_raise[env_idx] = max_d

    all_in_delta = player_stack - to_call
    exists_raise = (max_d >= min_d) or ((all_in_delta > 0) and (all_in_delta <= max_d))
    action_mask[env_idx, c.ACTION_RAISE] = exists_raise
