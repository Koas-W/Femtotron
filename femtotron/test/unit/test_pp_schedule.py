"""Unit tests for gpipe_schedule: pure-data action stream generation.

No GPU, no distributed, no model. Just verifies the schedule structure
for various (num_microbatches, is_first, is_last) combinations.
"""
from femtotron.parallel.pipeline_parallel.schedule import gpipe_schedule, one_f_one_b_schedule
from collections import defaultdict
from femtotron.parallel.pipeline_parallel.action import (
    PPAction,
    Forward, Backward,
    RecvForward, SendForward,
    RecvBackward, SendBackward,
    SendForwardRecvBackward, SendBackwardRecvForward,
)


# ──────────── helper: per-mb action accounting ────────────

def _per_mb_counts(actions, mb_id):
    """For mb_id, return (rf, f, sf, rb, b, sb) including combined-op contributions."""
    rf = f = sf = rb = b = sb = 0
    for a in actions:
        if   isinstance(a, Forward)        and a.mb_id == mb_id: f  += 1
        elif isinstance(a, Backward)       and a.mb_id == mb_id: b  += 1
        elif isinstance(a, RecvForward)    and a.mb_id == mb_id: rf += 1
        elif isinstance(a, SendForward)    and a.mb_id == mb_id: sf += 1
        elif isinstance(a, RecvBackward)   and a.mb_id == mb_id: rb += 1
        elif isinstance(a, SendBackward)   and a.mb_id == mb_id: sb += 1
        elif isinstance(a, SendForwardRecvBackward):
            if a.fwd_mb == mb_id: sf += 1
            if a.bwd_mb == mb_id: rb += 1
        elif isinstance(a, SendBackwardRecvForward):
            if a.bwd_mb == mb_id: sb += 1
            if a.fwd_mb == mb_id: rf += 1
    return rf, f, sf, rb, b, sb


def _check_per_mb_invariants(actions, num_mb, is_first, is_last, tag):
    """Each mb must have F=1, B=1, and the right number of each comm op for its role."""
    expected = {
        'f': 1, 'b': 1,
        'rf': 0 if is_first else 1,
        'sf': 0 if is_last  else 1,
        'rb': 0 if is_last  else 1,
        'sb': 0 if is_first else 1,
    }
    for mb_id in range(num_mb):
        rf, f, sf, rb, b, sb = _per_mb_counts(actions, mb_id)
        got = dict(rf=rf, f=f, sf=sf, rb=rb, b=b, sb=sb)
        for k, v in expected.items():
            assert got[k] == v, \
                f"{tag} mb={mb_id}: {k}={got[k]} (expected {v}); all={got}"


def _check_ordering(actions, num_mb, tag):
    """Per mb: RF < F < SF, RB < B < SB, F < B."""
    pos = defaultdict(dict)
    for i, a in enumerate(actions):
        if   isinstance(a, Forward):      pos['f'][a.mb_id]  = i
        elif isinstance(a, Backward):     pos['b'][a.mb_id]  = i
        elif isinstance(a, RecvForward):  pos['rf'][a.mb_id] = i
        elif isinstance(a, SendForward):  pos['sf'][a.mb_id] = i
        elif isinstance(a, RecvBackward): pos['rb'][a.mb_id] = i
        elif isinstance(a, SendBackward): pos['sb'][a.mb_id] = i
        elif isinstance(a, SendForwardRecvBackward):
            pos['sf'][a.fwd_mb] = i
            pos['rb'][a.bwd_mb] = i
        elif isinstance(a, SendBackwardRecvForward):
            pos['sb'][a.bwd_mb] = i
            pos['rf'][a.fwd_mb] = i

    for mb_id in range(num_mb):
        assert mb_id in pos['f'] and mb_id in pos['b'], \
            f"{tag} mb={mb_id}: missing F or B"
        assert pos['f'][mb_id] < pos['b'][mb_id], \
            f"{tag} mb={mb_id}: F@{pos['f'][mb_id]} >= B@{pos['b'][mb_id]}"
        for before, after in [('rf','f'), ('f','sf'), ('rb','b'), ('b','sb')]:
            if mb_id in pos[before] and mb_id in pos[after]:
                assert pos[before][mb_id] < pos[after][mb_id], \
                    f"{tag} mb={mb_id}: {before}@{pos[before][mb_id]} " \
                    f">= {after}@{pos[after][mb_id]}"

def log(msg):
    print(f"  {msg}")


def assert_eq_list(actual, expected, tag):
    if actual != expected:
        raise AssertionError(
            f"{tag}:\n  expected: {expected}\n  actual:   {actual}"
        )


def test_pp1_degenerate():
    """pp_size=1: both is_first and is_last. No comm actions, just F+B."""
    actions = gpipe_schedule(num_microbatches=3, is_first=True, is_last=True)
    expected = [Forward(0), Forward(1), Forward(2), Backward(2), Backward(1), Backward(0)]
    assert_eq_list(actions, expected, "pp1 N=3")
    log("✓ pp_size=1, N=3 ⇒ all F then reversed B, no comm")


def test_first_stage():
    """First stage: F then SF; later RB then B. No RF, no SB."""
    actions = gpipe_schedule(num_microbatches=2, is_first=True, is_last=False)
    expected = [
        Forward(0), SendForward(0), Forward(1), SendForward(1),
        RecvBackward(1), Backward(1), RecvBackward(0), Backward(0),
    ]
    assert_eq_list(actions, expected, "first stage N=2")
    log("✓ first stage, N=2 ⇒ F→SF then RB→B reversed, no RF/SB")


def test_last_stage():
    """Last stage: RF then F; later B then SB. No SF, no RB."""
    actions = gpipe_schedule(num_microbatches=2, is_first=False, is_last=True)
    expected = [
        RecvForward(0), Forward(0), RecvForward(1), Forward(1),
        Backward(1), SendBackward(1), Backward(0), SendBackward(0),
    ]
    assert_eq_list(actions, expected, "last stage N=2")
    log("✓ last stage, N=2 ⇒ RF→F then B→SB reversed, no SF/RB")


def test_mid_stage():
    """Mid stage: full triple RF→F→SF; later RB→B→SB."""
    actions = gpipe_schedule(num_microbatches=2, is_first=False, is_last=False)
    expected = [
        RecvForward(0), Forward(0), SendForward(0), RecvForward(1), Forward(1), SendForward(1),
        RecvBackward(1), Backward(1), SendBackward(1), RecvBackward(0), Backward(0), SendBackward(0),
    ]
    assert_eq_list(actions, expected, "mid stage N=2")
    log("✓ mid stage, N=2 ⇒ all six action types present")


def test_single_microbatch():
    """N=1 edge case: still valid, just one F and one B per stage."""
    for is_first, is_last, tag in [
        (True, True, "pp1 N=1"),
        (True, False, "first N=1"),
        (False, True, "last N=1"),
        (False, False, "mid N=1"),
    ]:
        actions = gpipe_schedule(1, is_first, is_last)
        # Should have at least F(0) and B(0)
        assert Forward(0) in actions, f"{tag}: missing Forward(0)"
        assert Backward(0) in actions, f"{tag}: missing Backward(0)"
        # B comes after F
        assert actions.index(Backward(0)) > actions.index(Forward(0)), \
            f"{tag}: Backward(0) should come after Forward(0)"
    log("✓ N=1 edge case works for all 4 role combinations")


def test_invalid_num_mb():
    """num_microbatches < 1 should raise."""
    try:
        gpipe_schedule(0, is_first=True, is_last=True)
    except ValueError as e:
        log(f"✓ rejected N=0 with ValueError: {e}")
        return
    raise AssertionError("Expected ValueError for N=0")


def test_action_count():
    """Sanity: action count matches expected formula."""
    # First stage: N forward (F+SF) + N backward (RB+B) = 4N
    # Mid stage: N forward (RF+F+SF) + N backward (RB+B+SB) = 6N
    # Last stage: N forward (RF+F) + N backward (B+SB) = 4N
    # pp1: N forward (F) + N backward (B) = 2N
    for N in [1, 2, 4, 8]:
        assert len(gpipe_schedule(N, True, True)) == 2 * N
        assert len(gpipe_schedule(N, True, False)) == 4 * N
        assert len(gpipe_schedule(N, False, True)) == 4 * N
        assert len(gpipe_schedule(N, False, False)) == 6 * N
    log("✓ action count matches formula for N ∈ {1,2,4,8}")


# ──────────── 1F1B tests ────────────

def test_1f1b_pp1_degenerate():
    """pp_size=1: F/B interleaved per mb, no comm."""
    actions = one_f_one_b_schedule(num_microbatches=3, pp_size=1, pp_rank=0)
    expected = [
        Forward(mb_id=0), Backward(mb_id=0),
        Forward(mb_id=1), Backward(mb_id=1),
        Forward(mb_id=2), Backward(mb_id=2),
    ]
    assert actions == expected
    log("✓ 1F1B pp_size=1: F/B interleaved per mb, no comm")


def test_1f1b_pp2_n4_first():
    """P=2 N=4 stage 0: SFRB in steady, RB in cool-down, no SBRF/RF."""
    actions = one_f_one_b_schedule(4, 2, 0)
    _check_per_mb_invariants(actions, 4, is_first=True, is_last=False, tag="P2N4S0")
    _check_ordering(actions, 4, "P2N4S0")
    sfrb = sum(1 for a in actions if isinstance(a, SendForwardRecvBackward))
    sbrf = sum(1 for a in actions if isinstance(a, SendBackwardRecvForward))
    assert sfrb == 3 and sbrf == 0
    assert len(actions) == 13
    log(f"✓ 1F1B P=2 N=4 stage 0: 13 actions ({sfrb} SFRB, {sbrf} SBRF)")


def test_1f1b_pp2_n4_last():
    """P=2 N=4 stage 1: SBRF in steady, RF first, plain SB on last steady."""
    actions = one_f_one_b_schedule(4, 2, 1)
    _check_per_mb_invariants(actions, 4, is_first=False, is_last=True, tag="P2N4S1")
    _check_ordering(actions, 4, "P2N4S1")
    sfrb = sum(1 for a in actions if isinstance(a, SendForwardRecvBackward))
    sbrf = sum(1 for a in actions if isinstance(a, SendBackwardRecvForward))
    assert sfrb == 0 and sbrf == 3
    assert len(actions) == 13
    log(f"✓ 1F1B P=2 N=4 stage 1: 13 actions ({sfrb} SFRB, {sbrf} SBRF)")


def test_1f1b_pp3_n4_mid():
    """P=3 N=4 stage 1 (mid): BOTH SFRB and SBRF should appear."""
    actions = one_f_one_b_schedule(4, 3, 1)
    _check_per_mb_invariants(actions, 4, is_first=False, is_last=False, tag="P3N4S1")
    _check_ordering(actions, 4, "P3N4S1")
    sfrb = sum(1 for a in actions if isinstance(a, SendForwardRecvBackward))
    sbrf = sum(1 for a in actions if isinstance(a, SendBackwardRecvForward))
    assert sfrb > 0 and sbrf > 0, f"mid stage should use both combined ops"
    log(f"✓ 1F1B P=3 N=4 stage 1 (mid): {len(actions)} actions, "
        f"{sfrb} SFRB, {sbrf} SBRF (both > 0 ✓)")


def test_1f1b_n_equals_p():
    """N == P: stage 0 has all-warmup, no steady. Should still be valid."""
    for pp_size in [2, 3, 4]:
        for pp_rank in range(pp_size):
            actions = one_f_one_b_schedule(pp_size, pp_size, pp_rank)
            tag = f"NeqP P{pp_size}R{pp_rank}"
            _check_per_mb_invariants(actions, pp_size,
                                     is_first=(pp_rank == 0),
                                     is_last=(pp_rank == pp_size - 1),
                                     tag=tag)
            _check_ordering(actions, pp_size, tag)
    log("✓ 1F1B N=P edge: invariants hold P ∈ {2,3,4}")


def test_1f1b_n_less_than_p():
    """N < P: warmup capped at N, num_steady=0 for some stages."""
    pp_size, num_mb = 4, 2
    for pp_rank in range(pp_size):
        actions = one_f_one_b_schedule(num_mb, pp_size, pp_rank)
        tag = f"NltP P{pp_size}R{pp_rank}"
        _check_per_mb_invariants(actions, num_mb,
                                 is_first=(pp_rank == 0),
                                 is_last=(pp_rank == pp_size - 1),
                                 tag=tag)
        _check_ordering(actions, num_mb, tag)
    log("✓ 1F1B N<P edge (P=4, N=2): invariants hold")


def test_1f1b_invalid_args():
    for kwargs in [
        dict(num_microbatches=0, pp_size=2, pp_rank=0),
        dict(num_microbatches=4, pp_size=0, pp_rank=0),
        dict(num_microbatches=4, pp_size=2, pp_rank=-1),
        dict(num_microbatches=4, pp_size=2, pp_rank=2),  # rank >= size
    ]:
        try:
            one_f_one_b_schedule(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"Expected ValueError for {kwargs}")
    log("✓ 1F1B invalid args rejected")


def test_1f1b_compute_count_invariant():
    """Cross-product sanity: every stage always does exactly N forwards and N backwards."""
    for num_mb in [1, 2, 4, 8]:
        for pp_size in [1, 2, 3, 4]:
            for pp_rank in range(pp_size):
                actions = one_f_one_b_schedule(num_mb, pp_size, pp_rank)
                f = sum(1 for a in actions if isinstance(a, Forward))
                b = sum(1 for a in actions if isinstance(a, Backward))
                assert f == num_mb, f"N={num_mb} P={pp_size} R={pp_rank}: F={f}"
                assert b == num_mb, f"N={num_mb} P={pp_size} R={pp_rank}: B={b}"
    log("✓ 1F1B compute count invariant: every stage has N F's and N B's")
    
def main():
    tests = [
        test_pp1_degenerate,
        test_first_stage,
        test_last_stage,
        test_mid_stage,
        test_single_microbatch,
        test_invalid_num_mb,
        test_action_count,
        # 1F1B tests (M4)
        test_1f1b_pp1_degenerate,
        test_1f1b_pp2_n4_first,
        test_1f1b_pp2_n4_last,
        test_1f1b_pp3_n4_mid,
        test_1f1b_n_equals_p,
        test_1f1b_n_less_than_p,
        test_1f1b_invalid_args,
        test_1f1b_compute_count_invariant,
    ]
    print(f"\nRunning {len(tests)} unit tests for gpipe_schedule\n")
    for t in tests:
        print(f"[{t.__name__}]")
        t()
    print(f"\n✅ All {len(tests)} tests passed\n")


if __name__ == "__main__":
    main()