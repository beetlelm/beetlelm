#!/usr/bin/env python3
"""
beetle_experiment.py — Experiment Builder CLI for BeetleLM

A higher-level interface than beetle_conditions.py for cognitive scientists
to design, sweep, and run bilingual/trilingual experiments.

Commands
--------
    # Interactively build an experiment with guided prompts
    python script/beetle_experiment.py design

    # Generate a single experiment config
    python script/beetle_experiment.py single B2 --l1 nld --l2 eng --ewc-lambda 20

    # Generate a full combinatorial sweep for MECO fitting
    python script/beetle_experiment.py sweep --meco --conditions B1 B2 B3 B4 B5

    # Generate sweep with custom grids
    python script/beetle_experiment.py sweep \
        --conditions B2 B3 \
        --pairs nld:eng deu:eng \
        --ewc 0 10 20 30 \
        --maml 0 0.2 0.4 \
        --lr-boost 1.0 2.0 3.0

    # Show what configs a sweep would generate (dry run)
    python script/beetle_experiment.py sweep --meco --conditions B2 --dry-run

    # List all conditions and their parameter recommendations
    python script/beetle_experiment.py guide
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def cmd_guide(args):
    """Print a cognitive scientist's guide to BeetleLM conditions."""
    from beetlelm.bilingual.data.condition_presets import CONDITION_DEFAULTS

    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║           BeetleLM Experiment Guide                        ║
    ║   Cognitively-Inspired Bilingual Language Modelling        ║
    ╚══════════════════════════════════════════════════════════════╝

    PARAMETER GUIDE
    ═══════════════

    EWC (Elastic Weight Consolidation)
    ───────────────────────────────────
    Controls L1 knowledge retention when learning L2.
      lambda = 0    No protection → model freely forgets L1
      lambda = 10   Mild retention → some L1 knowledge preserved
      lambda = 20   Moderate → balanced retention/plasticity
      lambda = 40   Strong → resists forgetting, slower L2 learning

    MAML (Model-Agnostic Meta-Learning)
    ─────────────────────────────────────
    Enables rapid adaptation to new language input.
      ratio = 0     Standard training (no meta-learning)
      ratio = 0.2   Mild meta-learning component
      ratio = 0.3   Moderate → faster language adaptation
      ratio = 0.5   Strong meta-learning

    LR Boost (Plasticity at Transitions)
    ──────────────────────────────────────
    How much faster the model learns at language transitions.
      boost = 1.0   No boost (constant learning rate)
      boost = 2.0   Doubles learning speed at transitions
      boost = 3.0   Triples learning speed (for late/sudden exposure)

    BILINGUAL CONDITIONS
    ════════════════════""")

    for name in ["B1", "B2", "B3", "B4", "B5"]:
        d = CONDITION_DEFAULTS[name]
        ewc = d.get("ewc", {})
        maml = d.get("maml", {})
        opt = d.get("optimization", {})
        exp_lr = opt.get("exposure_lr", {})

        ewc_str = f"lambda={ewc.get('lambda_', 0)}" if ewc.get("enabled") else "OFF"
        maml_str = f"ratio={maml.get('hybrid_ratio', 0)}" if maml.get("enabled") else "OFF"
        lr_str = opt.get("lr_scheduler", "linear")
        boost_str = f", boost={exp_lr.get('plasticity_boost', 1.0)}x" if exp_lr else ""

        print(f"""
    {name} — {d['description']}
      EWC:  {ewc_str}
      MAML: {maml_str}
      LR:   {lr_str}{boost_str}""")

    print("""
    TRILINGUAL CONDITIONS
    ═════════════════════""")

    for name in ["T1", "T2", "T3", "T4"]:
        d = CONDITION_DEFAULTS[name]
        ewc = d.get("ewc", {})
        maml = d.get("maml", {})
        opt = d.get("optimization", {})
        exp_lr = opt.get("exposure_lr", {})

        ewc_str = f"lambda={ewc.get('lambda_', 0)}" if ewc.get("enabled") else "OFF"
        maml_str = f"ratio={maml.get('hybrid_ratio', 0)}" if maml.get("enabled") else "OFF"
        lr_str = opt.get("lr_scheduler", "linear")
        boost_str = f", boost={exp_lr.get('plasticity_boost', 1.0)}x" if exp_lr else ""

        print(f"""
    {name} — {d['description']}
      EWC:  {ewc_str}
      MAML: {maml_str}
      LR:   {lr_str}{boost_str}""")

    print("""
    MECO FITTING WORKFLOW
    ═════════════════════
    1. Generate sweep:     python script/beetle_experiment.py sweep --meco --conditions B1 B2 B3 B4 B5
    2. Train all configs:  ./script/run_meco_sweep.sh <NODE_ID>
    3. Compute surprisal:  python script/compute_meco_surprisal.py --all
    4. Fit reading times:  Rscript ../beetlelm-rts-private/stats/meco_l2.Rmd
    """)


def cmd_single(args):
    """Generate a single experiment config."""
    from beetlelm.bilingual.data.experiment_builder import Experiment

    exp = Experiment(args.condition)

    if args.condition.upper().startswith("T"):
        if args.langs:
            exp.languages(langs=args.langs)
    else:
        exp.languages(l1=args.l1, l2=args.l2)

    if args.ewc_lambda is not None:
        exp.ewc(lambda_=args.ewc_lambda)
    if args.maml_ratio is not None:
        exp.maml(hybrid_ratio=args.maml_ratio)
    if args.lr_boost is not None:
        exp.lr(boost=args.lr_boost)
    if args.lr is not None:
        exp.lr(base=args.lr)

    cfg = exp.build()

    if args.output:
        path = cfg.to_yaml(args.output)
        print(f"Config written to: {path}")
    else:
        import yaml
        print(yaml.dump(cfg.to_dict(), sort_keys=False))

    print(f"\n{cfg.summary()}")


def cmd_sweep(args):
    """Generate a combinatorial sweep."""
    from beetlelm.bilingual.data.experiment_builder import Sweep

    conditions = [c.upper() for c in (args.conditions or [])]
    bilingual = [c for c in conditions if c.startswith("B")]
    trilingual = [c for c in conditions if c.startswith("T")]

    sweep = Sweep(bilingual=bilingual, trilingual=trilingual)

    if args.meco:
        sweep.meco_pairs_only()
    else:
        if args.pairs:
            pairs = [tuple(p.split(":")) for p in args.pairs]
            sweep.language_pairs(pairs)
        if args.triples:
            triples = [t.split(":") for t in args.triples]
            sweep.language_triples(triples)

    if args.ewc:
        sweep.ewc_grid([float(v) for v in args.ewc])
    if args.maml:
        sweep.maml_grid([float(v) for v in args.maml])
    if args.lr_boost:
        sweep.lr_boost_grid([float(v) for v in args.lr_boost])

    if args.dry_run:
        # Preview what would be generated
        print(f"\nDry Run — would generate to: {args.output}")
        print(sweep.summary())

        # Estimate config count
        from beetlelm.bilingual.data.condition_presets import CONDITION_DEFAULTS
        n = 0
        for c in bilingual:
            ewc_on = CONDITION_DEFAULTS.get(c, {}).get("ewc", {}).get("enabled", False)
            maml_on = CONDITION_DEFAULTS.get(c, {}).get("maml", {}).get("enabled", False)
            lr_on = CONDITION_DEFAULTS.get(c, {}).get("optimization", {}).get("lr_scheduler") == "exposure_based"
            n_ewc = len(args.ewc or [None]) if ewc_on and args.ewc else 1
            n_maml = len(args.maml or [None]) if maml_on and args.maml else 1
            n_lr = len(args.lr_boost or [None]) if lr_on and args.lr_boost else 1
            n_pairs = len(sweep._bi_pairs) if sweep._bi_pairs else 0
            n += n_pairs * n_ewc * n_maml * n_lr
        for c in trilingual:
            ewc_on = CONDITION_DEFAULTS.get(c, {}).get("ewc", {}).get("enabled", False)
            maml_on = CONDITION_DEFAULTS.get(c, {}).get("maml", {}).get("enabled", False)
            lr_on = CONDITION_DEFAULTS.get(c, {}).get("optimization", {}).get("lr_scheduler") == "exposure_based"
            n_ewc = len(args.ewc or [None]) if ewc_on and args.ewc else 1
            n_maml = len(args.maml or [None]) if maml_on and args.maml else 1
            n_lr = len(args.lr_boost or [None]) if lr_on and args.lr_boost else 1
            n_triples = len(sweep._tri_triples) if sweep._tri_triples else 0
            n += n_triples * n_ewc * n_maml * n_lr
        print(f"\nEstimated configs: {n}")
        return

    sweep.generate(args.output)
    print(f"\n{sweep.summary()}")
    print(f"\nConfigs written to: {args.output}")
    print(f"\nNext: run training with ./script/run_meco_sweep.sh <NODE_ID>")


def cmd_design(args):
    """Interactive experiment designer."""
    from beetlelm.bilingual.data.condition_presets import CONDITION_DEFAULTS, list_conditions
    from beetlelm.bilingual.data.experiment_builder import Experiment

    print("\n  BeetleLM Experiment Designer")
    print("  " + "=" * 40)
    print("\n  Available conditions:")
    for name, desc in list_conditions().items():
        print(f"    {name}: {desc}")

    condition = input("\n  Condition code (e.g., B2): ").strip().upper()
    if condition not in CONDITION_DEFAULTS:
        print(f"  Unknown condition: {condition}")
        return

    exp = Experiment(condition)
    defaults = CONDITION_DEFAULTS[condition]

    if condition.startswith("T"):
        langs_str = input("  Languages (3 codes, comma-separated, e.g., nld,eng,deu): ").strip()
        langs = [l.strip() for l in langs_str.split(",")]
        exp.languages(langs=langs)
    else:
        l1 = input("  L1 language code (e.g., nld): ").strip() or "eng"
        l2 = input("  L2 language code (e.g., eng): ").strip() or "nld"
        exp.languages(l1=l1, l2=l2)

    # EWC
    ewc_default = defaults.get("ewc", {})
    if ewc_default.get("enabled"):
        ewc_val = input(f"  EWC lambda [{ewc_default.get('lambda_', 0)}]: ").strip()
        if ewc_val:
            exp.ewc(lambda_=float(ewc_val))
    else:
        ewc_val = input(f"  EWC lambda [OFF, enter value to enable]: ").strip()
        if ewc_val:
            exp.ewc(lambda_=float(ewc_val))

    # MAML
    maml_default = defaults.get("maml", {})
    if maml_default.get("enabled"):
        maml_val = input(f"  MAML ratio [{maml_default.get('hybrid_ratio', 0)}]: ").strip()
        if maml_val:
            exp.maml(hybrid_ratio=float(maml_val))
    else:
        maml_val = input(f"  MAML ratio [OFF, enter value to enable]: ").strip()
        if maml_val:
            exp.maml(hybrid_ratio=float(maml_val))

    # LR
    opt = defaults.get("optimization", {})
    exp_lr = opt.get("exposure_lr", {})
    if exp_lr.get("plasticity_boost"):
        lr_val = input(f"  LR boost [{exp_lr['plasticity_boost']}x]: ").strip()
        if lr_val:
            exp.lr(boost=float(lr_val))

    cfg = exp.build()
    print(f"\n{cfg.summary()}")

    out_path = input(f"\n  Output path [configs/{cfg.run_name}.yaml]: ").strip()
    if not out_path:
        out_path = f"configs/{cfg.run_name}.yaml"

    path = cfg.to_yaml(out_path)
    print(f"\n  Config saved to: {path}")
    print(f"  Train with: python script/train.py --config_path {path}")


def main():
    parser = argparse.ArgumentParser(
        description="BeetleLM Experiment Builder for Cognitive Scientists",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script/beetle_experiment.py guide
  python script/beetle_experiment.py single B2 --l1 nld --l2 eng --ewc-lambda 20
  python script/beetle_experiment.py sweep --meco --conditions B1 B2 B3 B4 B5
  python script/beetle_experiment.py sweep --conditions B2 B3 --pairs nld:eng deu:eng --ewc 0 10 20
  python script/beetle_experiment.py design
        """,
    )
    subparsers = parser.add_subparsers(dest="command")

    # guide
    subparsers.add_parser("guide", help="Show parameter guide for cognitive scientists")

    # single
    p_single = subparsers.add_parser("single", help="Generate a single experiment config")
    p_single.add_argument("condition", help="Condition code (B1–B5, T1–T4)")
    p_single.add_argument("--l1", help="L1 language code")
    p_single.add_argument("--l2", help="L2 language code")
    p_single.add_argument("--langs", nargs="+", help="Language codes (trilingual)")
    p_single.add_argument("--ewc-lambda", type=float, help="EWC lambda (0=off)")
    p_single.add_argument("--maml-ratio", type=float, help="MAML hybrid ratio (0=off)")
    p_single.add_argument("--lr-boost", type=float, help="LR plasticity boost")
    p_single.add_argument("--lr", type=float, help="Base learning rate")
    p_single.add_argument("--output", "-o", help="Output YAML path")

    # sweep
    p_sweep = subparsers.add_parser("sweep", help="Generate combinatorial sweep configs")
    p_sweep.add_argument("--conditions", nargs="+", required=True, help="Conditions to sweep")
    p_sweep.add_argument("--meco", action="store_true", help="Use MECO language pairs")
    p_sweep.add_argument("--pairs", nargs="+", help="Bilingual pairs (l1:l2 format)")
    p_sweep.add_argument("--triples", nargs="+", help="Trilingual triples (l1:l2:l3 format)")
    p_sweep.add_argument("--ewc", nargs="+", help="EWC lambda values to sweep")
    p_sweep.add_argument("--maml", nargs="+", help="MAML ratio values to sweep")
    p_sweep.add_argument("--lr-boost", nargs="+", help="LR boost values to sweep")
    p_sweep.add_argument("--output", default="configs/conditions", help="Output directory")
    p_sweep.add_argument("--dry-run", action="store_true", help="Show what would be generated")

    # design
    subparsers.add_parser("design", help="Interactive experiment designer")

    args = parser.parse_args()
    commands = {
        "guide": cmd_guide,
        "single": cmd_single,
        "sweep": cmd_sweep,
        "design": cmd_design,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
