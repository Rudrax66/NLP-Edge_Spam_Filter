"""
main.py — Edge-Device Spam & Toxic Filter
Entry point: trains all models, runs comparison, shows results.

Usage:
  python main.py                    # full pipeline
  python main.py --model logistic   # single model
  python main.py --compare          # compare all models
  python main.py --test             # run test suite
"""

import sys
import os
import argparse
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import EdgeSpamToxicFilter
from data.dataset import generate_dataset


def compare_models(n_samples: int = 600):
    """Train and benchmark all available model types."""
    model_types = ["logistic", "svm", "naive_bayes", "random_forest", "ensemble"]
    texts, labels = generate_dataset(
        n_ham=n_samples // 3,
        n_spam=n_samples // 3,
        n_toxic=n_samples // 3
    )

    print("\n" + "="*65)
    print(f"  MODEL COMPARISON  ({len(texts)} samples)")
    print("="*65)
    print(f"  {'Model':<18} {'Accuracy':>10} {'F1 Score':>10} {'Train Time':>12}")
    print("-"*65)

    results = {}
    for mt in model_types:
        f = EdgeSpamToxicFilter(model_type=mt, verbose=False)
        t0 = time.perf_counter()
        m = f.train(texts=texts, labels=labels, n_samples=n_samples)
        elapsed = time.perf_counter() - t0
        results[mt] = {**m, "train_sec": elapsed}
        print(f"  {mt:<18} {m['accuracy']*100:>9.2f}% {m['f1_weighted']*100:>9.2f}% "
              f"{elapsed:>10.2f}s")

    best = max(results, key=lambda k: results[k]["f1_weighted"])
    print("-"*65)
    print(f"  Best model: {best.upper()} "
          f"(F1={results[best]['f1_weighted']*100:.2f}%)")
    print("="*65)
    return results


def run_quick_demo(model_type: str = "ensemble"):
    """Quick demonstration of the filter."""
    print("\n" + "─"*60)
    print("  QUICK DEMO")
    print("─"*60)

    sample_messages = [
        ("Hi, can we reschedule our call to Friday afternoon?", "HAM"),
        ("WINNER!!! You've been selected for a FREE $1000 gift card!! CLAIM NOW!!!", "SPAM"),
        ("You're pathetic and worthless. Nobody wants you around.", "TOXIC"),
        ("Please submit your timesheet by end of day Thursday.", "HAM"),
        ("Earn extra cash working from home! 100% GUARANTEED! Limited spots!", "SPAM"),
    ]

    f = EdgeSpamToxicFilter(model_type=model_type, verbose=True)
    f.train(n_samples=600)

    print("\n[Predictions]")
    correct = 0
    for text, expected in sample_messages:
        result = f.analyze(text)
        pred = result["prediction"]
        ok = pred == expected
        correct += int(ok)
        icon = "✓" if ok else "✗"
        print(f"  [{icon}] {pred:<6} ({result['confidence']*100:.1f}%) | {text[:55]}")

    print(f"\n  Accuracy: {correct}/{len(sample_messages)} "
          f"= {correct/len(sample_messages)*100:.0f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Edge-Device Spam & Toxic Filter"
    )
    parser.add_argument(
        "--model",
        choices=["logistic", "svm", "naive_bayes", "random_forest",
                 "gradient_boost", "ensemble"],
        default="ensemble",
        help="Model type to use (default: ensemble)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all model types"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run full test suite"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive demo"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=600,
        help="Number of training samples (default: 600)"
    )
    parser.add_argument(
        "--crossval",
        action="store_true",
        help="Run cross-validation during training"
    )

    args = parser.parse_args()

    print("\n" + "█"*60)
    print("  EDGE-DEVICE SPAM & TOXIC FILTER  |  Python NLP Project")
    print("  No API key required  •  Runs fully offline on device")
    print("█"*60)

    if args.test:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tests"))
        from tests.test_filter import run_all_tests
        run_all_tests()
        return

    if args.compare:
        compare_models(n_samples=args.samples)
        return

    if args.interactive:
        from demo import run_demo
        run_demo()
        return

    # Default: train + quick demo
    run_quick_demo(model_type=args.model)


if __name__ == "__main__":
    main()
