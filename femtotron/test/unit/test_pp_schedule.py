"""Unit tests for gpipe_schedule: pure-data action stream generation.

No GPU, no distributed, no model. Just verifies the schedule structure
for various (num_microbatches, is_first, is_last) combinations.
"""
from femtotron.parallel.pipeline_parallel.schedule import gpipe_schedule
from femtotron.parallel.pipeline_parallel.action  import PPAction, Forward, Backward, RecvForward, SendForward, RecvBackward, SendBackward, SendForwardRecvBackward


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


def main():
    tests = [
        test_pp1_degenerate,
        test_first_stage,
        test_last_stage,
        test_mid_stage,
        test_single_microbatch,
        test_invalid_num_mb,
        test_action_count,
    ]
    print(f"\nRunning {len(tests)} unit tests for gpipe_schedule\n")
    for t in tests:
        print(f"[{t.__name__}]")
        t()
    print(f"\n✅ All {len(tests)} tests passed\n")


if __name__ == "__main__":
    main()