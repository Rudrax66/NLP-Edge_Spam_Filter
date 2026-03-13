"""
Interactive CLI Demo — Edge-Device Spam & Toxic Filter
Run: python demo.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import EdgeSpamToxicFilter

BANNER = r"""
╔══════════════════════════════════════════════════════════╗
║     EDGE-DEVICE  SPAM  &  TOXIC  FILTER  (NLP)          ║
║     No API key  •  Runs 100% locally  •  Python          ║
╚══════════════════════════════════════════════════════════╝
"""

DEMO_MESSAGES = [
    "Can we schedule the team standup for Monday at 10 AM?",
    "CONGRATULATIONS!!! You've WON a FREE iPhone 15! CLICK HERE NOW!!!",
    "You're the most stupid and worthless person I've ever encountered.",
    "Please review the attached Q3 financial report before Friday.",
    "Earn $5,000/week from home!! No experience needed! LIMITED OFFER!",
    "I hate you. You should just disappear from everyone's life.",
    "The weather forecast shows rain for the next three days.",
    "Your PayPal account has been suspended. Verify now: http://paypa1.info",
    "Happy birthday! Hope you have a wonderful and relaxing day.",
    "Lose 30 lbs in 30 days! MIRACLE DIET PILL — 100% GUARANTEED!!!",
]


def run_demo():
    print(BANNER)

    # ── Train ──────────────────────────────────────────────────────────────────
    print("Step 1: Training the filter on synthetic data...")
    print("-" * 58)
    f = EdgeSpamToxicFilter(model_type="ensemble", verbose=True)
    metrics = f.train(n_samples=600, cross_validate=False)

    print(f"\n  ✓ Training complete!")
    print(f"  Accuracy : {metrics['accuracy']*100:.2f}%")
    print(f"  F1 Score : {metrics['f1_weighted']*100:.2f}%")

    # ── Demo messages ──────────────────────────────────────────────────────────
    print(f"\n\nStep 2: Analysing {len(DEMO_MESSAGES)} demo messages")
    print("=" * 58)

    for msg in DEMO_MESSAGES:
        result = f.analyze(msg)
        f.print_result(result)

    # ── Interactive mode ───────────────────────────────────────────────────────
    print("\n\nStep 3: Interactive Mode")
    print("  Type any message to classify it.")
    print("  Commands: 'quit' / 'exit' to stop, 'demo' to re-run demo")
    print("-" * 58)

    while True:
        try:
            user_input = input("\n📝 Enter message: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if user_input.lower() == "demo":
            for msg in DEMO_MESSAGES:
                f.print_result(f.analyze(msg))
            continue

        result = f.analyze(user_input)
        f.print_result(result)

    print("\n[Bye!]")


if __name__ == "__main__":
    run_demo()
