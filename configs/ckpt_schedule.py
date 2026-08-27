"""
configs/ckpt_schedule.py

Phase-aware log₂ checkpoint schedule for BabyBabel human-scale pretraining.

The key insight: checkpointing restarts at powers-of-two from *each* language
introduction point, so the model is densely checkpointed immediately after
every new language phase begins.

Usage
-----
    from configs.ckpt_schedule import phase_log2_schedule

    # Bilingual sequential (L2 at step 5000, max 10000)
    steps = phase_log2_schedule([0, 5000], max_steps=10_000)

    # Trilingual sequential (L2 at 3333, L3 at 6666)
    steps = phase_log2_schedule([0, 3333, 6666], max_steps=10_000)

    # No hard phase boundary (balanced, simultaneous)
    steps = phase_log2_schedule([0], max_steps=10_000)
"""


def phase_log2_schedule(
    phase_starts: list,
    max_steps: int,
    extra_offsets: list = None,
) -> list:
    """
    For each phase_start in phase_starts, emit a log₂ series:
      phase_start + 0, +1, +2, +4, +8, ..., +2^k  (while <= max_steps)

    Optionally adds extra linear offsets from each phase start for
    fine-grained checkpointing after phase transitions.

    Returns the sorted, deduplicated union of all series as a list of ints.

    Parameters
    ----------
    phase_starts : list[int]
        Step numbers at which a new language phase begins.
        Always include 0 for the initial phase.
    max_steps : int
        Maximum training step (inclusive).
    extra_offsets : list[int], optional
        Additional offsets from each phase start (e.g., [2, 4, 6, 8]).

    Examples
    --------
    >>> phase_log2_schedule([0, 5000], max_steps=10_000)
    [0, 1, 2, 4, 8, 16, ..., 4096, 5000, 5001, 5002, 5004, ...]

    >>> phase_log2_schedule([0, 5000], max_steps=10_000, extra_offsets=[2, 4, 6, 8])
    [0, 1, 2, 4, 6, 8, ..., 5000, 5001, 5002, 5004, 5006, 5008, ...]
    """
    steps = set()
    for start in phase_starts:
        # Always include the phase start itself
        if start <= max_steps:
            steps.add(start)
        # Powers of two offset from start
        k = 0
        while True:
            offset = 2 ** k
            step = start + offset
            if step > max_steps:
                break
            steps.add(step)
            k += 1
        # Extra offsets (e.g., for dense linear checkpointing after phase change)
        if extra_offsets:
            for off in extra_offsets:
                step = start + off
                if 0 <= step <= max_steps:
                    steps.add(step)
    return sorted(steps)


def fineweb_checkpoint_schedule(
    phase_starts: list,
    max_steps: int,
    subsample_log2: int = 2,
    extra_offsets=None,
) -> list:
    """
    Phase-aware log₂ checkpoint schedule for FineWeb-scale training.

    At 730K+ steps, a full log₂ series produces 40+ checkpoints per phase.
    This function subsamples by keeping every `subsample_log2`-th power of 2.

    Parameters
    ----------
    phase_starts : list[int]
        Step numbers at which a new language phase begins.
    max_steps : int
        Maximum training step (inclusive).
    subsample_log2 : int
        Keep every Nth power of 2 (default 2 = keep 1, 4, 16, 64, ...).
    extra_offsets : list[int], optional
        Additional offsets from each phase start (e.g., [2, 4, 6, 8] for
        fine-grained checkpointing after phase transitions).

    Returns
    -------
    Sorted, deduplicated list of checkpoint steps.
    """
    steps = set()
    for start in phase_starts:
        if start <= max_steps:
            steps.add(start)
        # Subsampled powers of two
        k = 0
        while True:
            offset = 2 ** k
            step = start + offset
            if step > max_steps:
                break
            if k % subsample_log2 == 0:  # keep every Nth power
                steps.add(step)
            k += 1
        # Extra offsets (e.g., for dense checkpointing after phase change)
        if extra_offsets:
            for off in extra_offsets:
                step = start + off
                if 0 <= step <= max_steps:
                    steps.add(step)
    return sorted(steps)


if __name__ == "__main__":
    # Verification printout
    print("=== Bilingual sequential (L2 at step 5000, max 10000) ===")
    s = phase_log2_schedule([0, 5000], max_steps=10_000)
    print(f"  {len(s)} checkpoints: {s}\n")

    print("=== Trilingual sequential (L2 at 3333, L3 at 6666, max 10000) ===")
    s = phase_log2_schedule([0, 3333, 6666], max_steps=10_000)
    print(f"  {len(s)} checkpoints: {s}\n")

    print("=== Bilingual balanced (no hard phase, max 10000) ===")
    s = phase_log2_schedule([0], max_steps=10_000)
    print(f"  {len(s)} checkpoints: {s}\n")

    print("=== Bilingual late (L2 at step 8000, max 10000) ===")
    s = phase_log2_schedule([0, 8000], max_steps=10_000)
    print(f"  {len(s)} checkpoints: {s}\n")

    print("=== FineWeb bilingual balanced (L2 at 365000, max 730000) ===")
    s = fineweb_checkpoint_schedule([0, 365_000], max_steps=730_000)
    print(f"  {len(s)} checkpoints: {s[:10]}...{s[-5:]}\n")

    print("=== FineWeb bilingual balanced with extra offsets ===")
    s = fineweb_checkpoint_schedule([0, 365_000], max_steps=730_000,
                                     extra_offsets=[2, 4, 6, 8])
    print(f"  {len(s)} checkpoints: {s[:15]}...{s[-5:]}\n")
