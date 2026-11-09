import warp as wp

from gpu_poker import constants as c
from gpu_poker.kernels import actions
from gpu_poker.struct_types import GameState


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

    min_d = actions.min_raise_delta(state, env_idx)
    max_d = actions.max_raise_delta(state, env_idx, player)

    min_raise[env_idx] = min_d
    max_raise[env_idx] = max_d

    all_in_delta = player_stack - to_call
    exists_raise = (max_d >= min_d) or ((all_in_delta > 0) and (all_in_delta <= max_d))
    action_mask[env_idx, c.ACTION_RAISE] = exists_raise
