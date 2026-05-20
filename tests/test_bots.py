import torch
from train import bots

from gpu_poker import constants as c


def test_calling_station_respects_mask():
    mask = torch.zeros((4, 4), dtype=torch.bool)
    mask[:, c.ACTION_FOLD] = True
    mask[0, c.ACTION_CHECK] = True
    mask[1, c.ACTION_CALL] = True
    mask[2, c.ACTION_RAISE] = True
    # env 3: only fold

    a, rf = bots.calling_station({"action_mask": mask})
    assert a.dtype == torch.int32
    assert rf.shape == (4, 1)
    assert int(a[0].item()) == c.ACTION_CHECK
    assert int(a[1].item()) == c.ACTION_CALL
    assert int(a[2].item()) == c.ACTION_RAISE
    assert int(a[3].item()) == c.ACTION_FOLD


def test_random_aggressive_always_legal():
    torch.manual_seed(0)
    mask = torch.zeros((128, 4), dtype=torch.bool)
    mask[:, c.ACTION_FOLD] = True
    mask[:, c.ACTION_CHECK] = True
    mask[:, c.ACTION_CALL] = True
    mask[:, c.ACTION_RAISE] = True
    a, rf = bots.random_aggressive({"action_mask": mask})
    assert a.shape == (128,)
    assert rf.shape == (128, 1)
    # Check legality
    assert torch.all(mask[torch.arange(128), a.to(dtype=torch.int64)])


def test_random_aggressive_respects_requested_mixture_when_all_actions_legal():
    torch.manual_seed(0)
    mask = torch.ones((4096, 4), dtype=torch.bool)

    a, _ = bots.random_aggressive(
        {"action_mask": mask},
        p_raise=0.5,
        p_call=0.4,
        p_fold=0.1,
    )

    assert a.eq(c.ACTION_RAISE).any()
    assert a.eq(c.ACTION_CALL).any()
    assert a.eq(c.ACTION_FOLD).any()
    assert not torch.all(a.eq(c.ACTION_FOLD))


def test_other_bots_always_legal():
    mask = torch.zeros((256, 4), dtype=torch.bool)
    mask[:, c.ACTION_FOLD] = True
    mask[:, c.ACTION_CHECK] = True
    mask[:, c.ACTION_CALL] = True
    mask[:, c.ACTION_RAISE] = True
    for fn in (bots.nit, bots.loose_passive, bots.loose_aggressive):
        a, rf = fn({"action_mask": mask})
        assert a.shape == (256,)
        assert rf.shape == (256, 1)
        assert torch.all(mask[torch.arange(256), a.to(dtype=torch.int64)])


def _seeded_generator(seed: int) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g


def test_random_aggressive_deterministic_with_seeded_generator():
    """Without a generator, repeated calls draw from the global RNG and are
    non-reproducible. Passing an explicit Generator must yield bit-identical
    action selection and raise_frac across two invocations."""
    mask = torch.ones((512, 4), dtype=torch.bool)

    a1, rf1 = bots.random_aggressive({"action_mask": mask}, generator=_seeded_generator(42))
    a2, rf2 = bots.random_aggressive({"action_mask": mask}, generator=_seeded_generator(42))
    assert torch.equal(a1, a2)
    assert torch.equal(rf1, rf2)

    # Different seed -> some differences expected.
    a3, _ = bots.random_aggressive({"action_mask": mask}, generator=_seeded_generator(7))
    assert not torch.equal(a1, a3)


def test_loose_aggressive_deterministic_with_seeded_generator():
    mask = torch.ones((512, 4), dtype=torch.bool)

    a1, rf1 = bots.loose_aggressive({"action_mask": mask}, generator=_seeded_generator(123))
    a2, rf2 = bots.loose_aggressive({"action_mask": mask}, generator=_seeded_generator(123))
    assert torch.equal(a1, a2)
    assert torch.equal(rf1, rf2)


def test_get_bot_fn_propagates_generator():
    """The wrapper closures from `get_bot_fn` must use the passed-in generator
    so eval seeding works end-to-end."""
    mask = torch.ones((256, 4), dtype=torch.bool)

    g1 = _seeded_generator(999)
    bot1 = bots.get_bot_fn("random_aggressive", generator=g1)
    a1, _ = bot1({"action_mask": mask})

    g2 = _seeded_generator(999)
    bot2 = bots.get_bot_fn("random_aggressive", generator=g2)
    a2, _ = bot2({"action_mask": mask})

    assert torch.equal(a1, a2)
