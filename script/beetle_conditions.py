#!/usr/bin/env python3
"""
beetle_conditions.py — Cognitive Scientist Interface for BeetleLM Conditions

A command-line tool for cognitive scientists to explore, configure, and run
bilingual/trilingual training conditions without editing YAML directly.

Usage
-----
    # List all available conditions with descriptions
    python script/beetle_conditions.py list

    # Show detailed info about a condition (exposure schedule, recommended settings)
    python script/beetle_conditions.py info B3

    # Generate a config with defaults
    python script/beetle_conditions.py create B2 --l1 nld --l2 eng

    # Generate a config with custom EWC/MAML/LR overrides
    python script/beetle_conditions.py create B2 --l1 nld --l2 eng \
        --ewc-lambda 30 --lr-boost 2.5

    # Generate a trilingual config
    python script/beetle_conditions.py create T2 --langs nld eng deu

    # Generate a full sweep for MECO fitting
    python script/beetle_conditions.py sweep --meco-only

    # Visualise the exposure schedule for a condition
    python script/beetle_conditions.py plot B3 --steps 10000

    # Show what a condition does at specific training progress
    python script/beetle_conditions.py probe B2 --at 0.3 0.5 0.7 0.9
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from beetlelm.bilingual.data.condition_presets import (
    get_condition,
    get_condition_defaults,
    list_conditions,
    CONDITION_DEFAULTS,
    BILINGUAL_CONDITION_BUILDERS,
    TRILINGUAL_CONDITION_BUILDERS,
)
from beetlelm.bilingual.data.smooth_curriculum import normalise_weights


def cmd_list(args):
    """List all available conditions."""
    print("\n  BeetleLM Training Conditions")
    print("  " + "=" * 50)

    print("\n  BILINGUAL (B1–B5)")
    print("  " + "-" * 40)
    for name in sorted(BILINGUAL_CONDITION_BUILDERS):
        desc = CONDITION_DEFAULTS[name]["description"]
        defaults = CONDITION_DEFAULTS[name]
        ewc = "EWC" if defaults.get("ewc", {}).get("enabled") else "   "
        maml = "MAML" if defaults.get("maml", {}).get("enabled") else "    "
        lr = defaults.get("optimization", {}).get("lr_scheduler", "linear")
        lr_short = "exp-LR" if "exposure" in lr else "lin-LR"
        print(f"  {name}  {desc:<45s}  [{ewc}] [{maml}] [{lr_short}]")

    print("\n  TRILINGUAL (T1–T4)")
    print("  " + "-" * 40)
    for name in sorted(TRILINGUAL_CONDITION_BUILDERS):
        desc = CONDITION_DEFAULTS[name]["description"]
        defaults = CONDITION_DEFAULTS[name]
        ewc = "EWC" if defaults.get("ewc", {}).get("enabled") else "   "
        maml = "MAML" if defaults.get("maml", {}).get("enabled") else "    "
        lr = defaults.get("optimization", {}).get("lr_scheduler", "linear")
        lr_short = "exp-LR" if "exposure" in lr else "lin-LR"
        print(f"  {name}  {desc:<45s}  [{ewc}] [{maml}] [{lr_short}]")

    print()


def cmd_info(args):
    """Show detailed info about a condition."""
    name = args.condition.upper()
    if name not in CONDITION_DEFAULTS:
        print(f"Unknown condition: {name}")
        print(f"Available: {', '.join(sorted(CONDITION_DEFAULTS))}")
        return

    defaults = get_condition_defaults(name)
    print(f"\n  Condition: {name} — {defaults['description']}")
    print("  " + "=" * 60)

    # EWC
    ewc = defaults.get("ewc", {})
    if ewc.get("enabled"):
        print(f"\n  EWC (Elastic Weight Consolidation):")
        print(f"    lambda     = {ewc.get('lambda_', 'N/A')}")
        print(f"    fisher_n   = {ewc.get('n_fisher_samples', 'N/A')}")
        print(f"    trigger    = {ewc.get('trigger', 'phase_change')}")
    else:
        print(f"\n  EWC: disabled")

    # MAML
    maml = defaults.get("maml", {})
    if maml.get("enabled"):
        print(f"\n  MAML (Meta-Learning):")
        print(f"    hybrid_ratio = {maml.get('hybrid_ratio', 'N/A')}")
        print(f"    n_way        = {maml.get('n_way', 5)}")
        print(f"    k_shot       = {maml.get('k_shot', 1)}")
        print(f"    inner_steps  = {maml.get('inner_steps', 3)}")
        print(f"    inner_lr     = {maml.get('inner_lr', 0.1)}")
    else:
        print(f"\n  MAML: disabled")

    # LR
    opt = defaults.get("optimization", {})
    print(f"\n  Learning Rate:")
    print(f"    base_lr    = {opt.get('lr', 5e-4)}")
    print(f"    scheduler  = {opt.get('lr_scheduler', 'linear_with_warmup')}")
    if "exposure_lr" in opt:
        exp = opt["exposure_lr"]
        print(f"    boost      = {exp.get('plasticity_boost', 1.0)}x at phase changes")
        print(f"    warmup     = {exp.get('warmup_tokens', 'N/A'):,} tokens")
        print(f"    decay      = {exp.get('decay_shape', 'cosine')}")

    # Exposure schedule preview
    print(f"\n  Exposure Schedule Preview:")
    total_steps = args.steps if hasattr(args, 'steps') and args.steps else 10_000
    try:
        if name.startswith("T"):
            mix = get_condition(name, total_steps=total_steps, lang_keys=["l1", "l2", "l3"])
        else:
            mix = get_condition(name, total_steps=total_steps)

        for t in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            w = normalise_weights(mix.schedule.weights_at(t))
            bars = "  ".join(f"{k}={v:.2f}" for k, v in sorted(w.items()))
            print(f"    t={t:.2f}: {bars}")
    except Exception as e:
        print(f"    (could not preview: {e})")

    print()


def cmd_probe(args):
    """Show exposure weights at specific progress points."""
    name = args.condition.upper()
    total_steps = args.steps or 10_000
    points = args.at or [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    try:
        if name.startswith("T"):
            langs = args.langs or ["l1", "l2", "l3"]
            mix = get_condition(name, total_steps=total_steps, lang_keys=langs)
        else:
            mix = get_condition(name, total_steps=total_steps)
    except Exception as e:
        print(f"Error: {e}")
        return

    print(f"\n  {name} exposure at {total_steps:,} steps:")
    print("  " + "-" * 50)
    for t in points:
        w = normalise_weights(mix.schedule.weights_at(t))
        step = int(t * total_steps)
        bars = "  ".join(f"{k}:{v:.3f}" for k, v in sorted(w.items()))
        # Visual bar
        visual = ""
        for k, v in sorted(w.items()):
            n_blocks = int(v * 30)
            visual += f"{'#' * n_blocks}"
            visual += " "
        print(f"  step {step:>6d} (t={t:.2f}): {bars}  |{visual}|")
    print()


def cmd_create(args):
    """Generate a YAML config for a condition."""
    name = args.condition.upper()
    if name not in CONDITION_DEFAULTS:
        print(f"Unknown condition: {name}. Available: {', '.join(sorted(CONDITION_DEFAULTS))}")
        return

    defaults = get_condition_defaults(name)

    # Apply overrides
    if args.ewc_lambda is not None:
        if "ewc" not in defaults:
            defaults["ewc"] = {}
        defaults["ewc"]["enabled"] = args.ewc_lambda > 0
        defaults["ewc"]["lambda_"] = float(args.ewc_lambda)

    if args.maml_ratio is not None:
        if "maml" not in defaults:
            defaults["maml"] = {}
        defaults["maml"]["enabled"] = args.maml_ratio > 0
        defaults["maml"]["hybrid_ratio"] = float(args.maml_ratio)

    if args.lr_boost is not None:
        if "optimization" not in defaults:
            defaults["optimization"] = {}
        if "exposure_lr" not in defaults["optimization"]:
            defaults["optimization"]["exposure_lr"] = {}
        defaults["optimization"]["exposure_lr"]["plasticity_boost"] = float(args.lr_boost)

    if args.lr is not None:
        defaults["optimization"]["lr"] = float(args.lr)

    # Build the config
    from configs.generate_conditions import _make_bilingual_config, _make_trilingual_config

    if name.startswith("B"):
        l1 = args.l1 or "eng"
        l2 = args.l2 or "nld"
        suffix = ""
        if args.ewc_lambda is not None:
            suffix += f"_ewc{args.ewc_lambda}"
        if args.maml_ratio is not None:
            suffix += f"_maml{args.maml_ratio}"
        if args.lr_boost is not None:
            suffix += f"_lrboost{args.lr_boost}"

        cfg = _make_bilingual_config(
            name, l1, l2,
            ewc_override=defaults.get("ewc"),
            maml_override=defaults.get("maml"),
            lr_override=defaults.get("optimization"),
            suffix=suffix,
        )
    elif name.startswith("T"):
        langs = args.langs or ["eng", "nld", "deu"]
        suffix = ""
        if args.ewc_lambda is not None:
            suffix += f"_ewc{args.ewc_lambda}"
        if args.maml_ratio is not None:
            suffix += f"_maml{args.maml_ratio}"

        cfg = _make_trilingual_config(
            name, langs,
            ewc_override=defaults.get("ewc"),
            maml_override=defaults.get("maml"),
            lr_override=defaults.get("optimization"),
            suffix=suffix,
        )
    else:
        print(f"Unsupported condition prefix: {name}")
        return

    # Output
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            yaml.dump(cfg, f, sort_keys=False)
        print(f"Config written to: {out_path}")
    else:
        print(yaml.dump(cfg, sort_keys=False))


def cmd_sweep(args):
    """Generate all sweep configs."""
    from configs.generate_conditions import generate
    generate()


def cmd_plot(args):
    """Plot the exposure schedule for a condition (requires matplotlib)."""
    name = args.condition.upper()
    total_steps = args.steps or 10_000

    try:
        if name.startswith("T"):
            langs = args.langs or ["l1", "l2", "l3"]
            mix = get_condition(name, total_steps=total_steps, lang_keys=langs)
        else:
            mix = get_condition(name, total_steps=total_steps)
    except Exception as e:
        print(f"Error: {e}")
        return

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib required for plotting. Install: pip install matplotlib")
        return

    ts = np.linspace(0, 1, 500)
    all_weights = [normalise_weights(mix.schedule.weights_at(float(t))) for t in ts]
    langs = sorted(all_weights[0].keys())

    fig, ax = plt.subplots(figsize=(10, 5))
    for lang in langs:
        values = [w.get(lang, 0) for w in all_weights]
        ax.plot(ts * total_steps, values, label=lang, linewidth=2)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Exposure Weight")
    ax.set_title(f"{name}: {CONDITION_DEFAULTS[name]['description']}")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    out_path = args.output or f"{name}_exposure.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to: {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="BeetleLM Condition Interface for Cognitive Scientists",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script/beetle_conditions.py list
  python script/beetle_conditions.py info B3
  python script/beetle_conditions.py probe B2 --at 0.0 0.25 0.5 0.75 1.0
  python script/beetle_conditions.py create B2 --l1 nld --l2 eng --ewc-lambda 20
  python script/beetle_conditions.py create T2 --langs nld eng deu
  python script/beetle_conditions.py plot B3 --steps 10000
  python script/beetle_conditions.py sweep
        """,
    )
    subparsers = parser.add_subparsers(dest="command")

    # list
    subparsers.add_parser("list", help="List all available conditions")

    # info
    p_info = subparsers.add_parser("info", help="Show detailed info about a condition")
    p_info.add_argument("condition", help="Condition code (B1–B5, T1–T4)")
    p_info.add_argument("--steps", type=int, default=10_000)

    # probe
    p_probe = subparsers.add_parser("probe", help="Show exposure weights at specific points")
    p_probe.add_argument("condition", help="Condition code")
    p_probe.add_argument("--at", type=float, nargs="+", help="Progress points (0–1)")
    p_probe.add_argument("--steps", type=int, default=10_000)
    p_probe.add_argument("--langs", nargs="+", help="Language keys (for trilingual)")

    # create
    p_create = subparsers.add_parser("create", help="Generate a YAML config")
    p_create.add_argument("condition", help="Condition code")
    p_create.add_argument("--l1", help="L1 language code")
    p_create.add_argument("--l2", help="L2 language code")
    p_create.add_argument("--langs", nargs="+", help="Language keys (for trilingual)")
    p_create.add_argument("--ewc-lambda", type=float, help="Override EWC lambda")
    p_create.add_argument("--maml-ratio", type=float, help="Override MAML hybrid_ratio")
    p_create.add_argument("--lr-boost", type=float, help="Override LR plasticity_boost")
    p_create.add_argument("--lr", type=float, help="Override base learning rate")
    p_create.add_argument("--output", "-o", help="Output YAML path (stdout if omitted)")

    # sweep
    p_sweep = subparsers.add_parser("sweep", help="Generate all sweep configs")
    p_sweep.add_argument("--meco-only", action="store_true", help="Only MECO-relevant pairs")

    # plot
    p_plot = subparsers.add_parser("plot", help="Plot exposure schedule")
    p_plot.add_argument("condition", help="Condition code")
    p_plot.add_argument("--steps", type=int, default=10_000)
    p_plot.add_argument("--langs", nargs="+", help="Language keys (for trilingual)")
    p_plot.add_argument("--output", "-o", help="Output image path")

    args = parser.parse_args()

    commands = {
        "list": cmd_list,
        "info": cmd_info,
        "probe": cmd_probe,
        "create": cmd_create,
        "sweep": cmd_sweep,
        "plot": cmd_plot,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
