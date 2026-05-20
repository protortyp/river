from __future__ import annotations

import torch

from gpu_poker import constants as c


def calling_station(obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Simple baseline: never raises unless forced by legality.

    Policy:
    - CHECK if legal
    - else CALL if legal
    - else RAISE with min-raise (raise_frac=0) if legal
    - else FOLD

    Returns:
        action_type: int32 [N]
        raise_frac: float32 [N,1]
    """
    mask = obs["action_mask"]  # [N,4] bool
    n = mask.shape[0]
    device = mask.device

    action = torch.full((n,), c.ACTION_FOLD, device=device, dtype=torch.int32)

    can_check = mask[:, c.ACTION_CHECK]
    can_call = mask[:, c.ACTION_CALL]
    can_raise = mask[:, c.ACTION_RAISE]

    action = torch.where(
        can_raise, torch.tensor(c.ACTION_RAISE, device=device, dtype=torch.int32), action
    )
    action = torch.where(
        can_call, torch.tensor(c.ACTION_CALL, device=device, dtype=torch.int32), action
    )
    action = torch.where(
        can_check, torch.tensor(c.ACTION_CHECK, device=device, dtype=torch.int32), action
    )

    raise_frac = torch.zeros((n, 1), device=device, dtype=torch.float32)
    return action, raise_frac


def get_bot_fn(bot_name: str, *, generator: torch.Generator | None = None):
    """Return a bot closure. `generator` seeds stochastic bots reproducibly.

    Deterministic bots (calling_station, nit, loose_passive) ignore `generator`.
    Stochastic bots (random_aggressive, loose_aggressive) use it for every
    `torch.rand` call so that the same eval run is bit-reproducible.
    """
    if bot_name == "calling_station":
        return calling_station
    if bot_name == "random_aggressive":

        def bot_fn(o: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
            return random_aggressive(o, p_raise=0.5, p_call=0.4, p_fold=0.1, generator=generator)

        return bot_fn
    if bot_name == "nit":
        return nit
    if bot_name == "loose_passive":
        return loose_passive
    if bot_name == "loose_aggressive":

        def bot_fn(o: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
            return loose_aggressive(o, generator=generator)

        return bot_fn
    raise ValueError(f"Unknown bot_name: {bot_name}")


def available_bot_names() -> tuple[str, ...]:
    # Keep this stable and explicit (useful for eval history comparisons).
    return (
        "calling_station",
        "random_aggressive",
        "nit",
        "loose_passive",
        "loose_aggressive",
    )


def _fallback_legal_action(mask: torch.Tensor) -> torch.Tensor:
    # Deterministic fallback: CHECK > CALL > RAISE > FOLD.
    device = mask.device
    n = mask.shape[0]
    action = torch.full((n,), c.ACTION_FOLD, device=device, dtype=torch.int32)
    # Apply lower priority first, then overwrite with higher priority.
    action = torch.where(
        mask[:, c.ACTION_RAISE],
        torch.tensor(c.ACTION_RAISE, device=device, dtype=torch.int32),
        action,
    )
    action = torch.where(
        mask[:, c.ACTION_CALL],
        torch.tensor(c.ACTION_CALL, device=device, dtype=torch.int32),
        action,
    )
    action = torch.where(
        mask[:, c.ACTION_CHECK],
        torch.tensor(c.ACTION_CHECK, device=device, dtype=torch.int32),
        action,
    )
    return action


def nit(obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Very tight / risk-averse baseline:
    - CHECK if legal
    - else FOLD if legal
    - else CALL
    - else RAISE min (should be rare)
    """
    mask = obs["action_mask"]  # [N,4] bool
    n = mask.shape[0]
    device = mask.device

    action = torch.full((n,), c.ACTION_FOLD, device=device, dtype=torch.int32)
    can_check = mask[:, c.ACTION_CHECK]
    can_fold = mask[:, c.ACTION_FOLD]
    can_call = mask[:, c.ACTION_CALL]
    can_raise = mask[:, c.ACTION_RAISE]

    action = torch.where(
        can_raise, torch.tensor(c.ACTION_RAISE, device=device, dtype=torch.int32), action
    )
    action = torch.where(
        can_call, torch.tensor(c.ACTION_CALL, device=device, dtype=torch.int32), action
    )
    action = torch.where(
        can_fold, torch.tensor(c.ACTION_FOLD, device=device, dtype=torch.int32), action
    )
    action = torch.where(
        can_check, torch.tensor(c.ACTION_CHECK, device=device, dtype=torch.int32), action
    )

    # Ensure legality if mask has surprising patterns.
    legal = mask[torch.arange(n, device=device), action.to(dtype=torch.int64)]
    if (~legal).any():
        action = torch.where(~legal, _fallback_legal_action(mask), action)

    raise_frac = torch.zeros((n, 1), device=device, dtype=torch.float32)
    return action, raise_frac


def loose_passive(obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Loose-passive baseline:
    - CALL if legal else CHECK if legal
    - almost never raises (only if forced by legality)
    """
    mask = obs["action_mask"]  # [N,4] bool
    n = mask.shape[0]
    device = mask.device

    action = torch.full((n,), c.ACTION_CHECK, device=device, dtype=torch.int32)
    action = torch.where(
        mask[:, c.ACTION_CALL],
        torch.tensor(c.ACTION_CALL, device=device, dtype=torch.int32),
        action,
    )
    action = torch.where(
        (~mask[:, c.ACTION_CALL]) & mask[:, c.ACTION_CHECK],
        torch.tensor(c.ACTION_CHECK, device=device, dtype=torch.int32),
        action,
    )
    # If neither check nor call is legal, fall back (may include raise).
    legal = mask[torch.arange(n, device=device), action.to(dtype=torch.int64)]
    if (~legal).any():
        action = torch.where(~legal, _fallback_legal_action(mask), action)

    raise_frac = torch.zeros((n, 1), device=device, dtype=torch.float32)
    return action, raise_frac


def loose_aggressive(
    obs: dict[str, torch.Tensor],
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Loose-aggressive baseline:
    - RAISE if legal (random raise_frac)
    - else CALL if legal
    - else CHECK
    - else FOLD
    """
    mask = obs["action_mask"]  # [N,4] bool
    n = mask.shape[0]
    device = mask.device

    action = torch.full((n,), c.ACTION_CHECK, device=device, dtype=torch.int32)
    action = torch.where(
        mask[:, c.ACTION_RAISE],
        torch.tensor(c.ACTION_RAISE, device=device, dtype=torch.int32),
        action,
    )
    action = torch.where(
        (~mask[:, c.ACTION_RAISE]) & mask[:, c.ACTION_CALL],
        torch.tensor(c.ACTION_CALL, device=device, dtype=torch.int32),
        action,
    )
    action = torch.where(
        (~mask[:, c.ACTION_RAISE]) & (~mask[:, c.ACTION_CALL]) & mask[:, c.ACTION_CHECK],
        torch.tensor(c.ACTION_CHECK, device=device, dtype=torch.int32),
        action,
    )
    action = torch.where(
        (~mask[:, c.ACTION_RAISE])
        & (~mask[:, c.ACTION_CALL])
        & (~mask[:, c.ACTION_CHECK])
        & mask[:, c.ACTION_FOLD],
        torch.tensor(c.ACTION_FOLD, device=device, dtype=torch.int32),
        action,
    )

    # Ensure legality if needed.
    legal = mask[torch.arange(n, device=device), action.to(dtype=torch.int64)]
    if (~legal).any():
        action = torch.where(~legal, _fallback_legal_action(mask), action)

    raise_frac = torch.zeros((n, 1), device=device, dtype=torch.float32)
    rf = torch.rand((n, 1), device=device, dtype=torch.float32, generator=generator)
    raise_frac = torch.where(action.view(-1, 1).eq(c.ACTION_RAISE), rf, raise_frac)
    return action, raise_frac


def random_aggressive(
    obs: dict[str, torch.Tensor],
    *,
    p_raise: float = 0.5,
    p_call: float = 0.4,
    p_fold: float = 0.1,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Random aggressive baseline:
    - with prob p_raise: raise if legal (uniform raise_frac), else fall back
    - with prob p_call: check/call if legal
    - with prob p_fold: fold if legal
    """
    mask = obs["action_mask"]  # [N,4] bool
    n = mask.shape[0]
    device = mask.device

    u = torch.rand((n,), device=device, generator=generator)

    want_raise = u < p_raise
    want_call = (u >= p_raise) & (u < p_raise + p_call)
    want_fold = u >= (p_raise + p_call)

    fallback = _fallback_legal_action(mask)
    action = fallback

    # Fold selection (if legal; else later fallbacks will overwrite).
    action = torch.where(
        want_fold & mask[:, c.ACTION_FOLD],
        torch.tensor(c.ACTION_FOLD, device=device, dtype=torch.int32),
        action,
    )

    # Call/check selection.
    action = torch.where(
        want_call & mask[:, c.ACTION_CALL],
        torch.tensor(c.ACTION_CALL, device=device, dtype=torch.int32),
        action,
    )
    action = torch.where(
        want_call & (~mask[:, c.ACTION_CALL]) & mask[:, c.ACTION_CHECK],
        torch.tensor(c.ACTION_CHECK, device=device, dtype=torch.int32),
        action,
    )

    # Raise selection (uniform raise_frac).
    action = torch.where(
        want_raise & mask[:, c.ACTION_RAISE],
        torch.tensor(c.ACTION_RAISE, device=device, dtype=torch.int32),
        action,
    )

    legal = mask[torch.arange(n, device=device), action.to(dtype=torch.int64)]
    action = torch.where(legal, action, fallback)

    # Raise frac: random only when raising, else 0.
    raise_frac = torch.zeros((n, 1), device=device, dtype=torch.float32)
    rf = torch.rand((n, 1), device=device, dtype=torch.float32, generator=generator)
    raise_frac = torch.where(action.view(-1, 1).eq(c.ACTION_RAISE), rf, raise_frac)

    # Normalize weights for future extensibility; keep signature stable.
    _ = (p_fold,)  # noqa: F841
    return action, raise_frac
