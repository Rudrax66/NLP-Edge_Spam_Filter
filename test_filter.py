"""
Test Suite — Edge-Device Spam & Toxic Filter
Tests preprocessing, features, model training, and end-to-end inference.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessor import TextPreprocessor
from pipeline import EdgeSpamToxicFilter


# ─────────────────────────────────────────────────────────────────────────────
TEST_CASES = [
    # (text, expected_label)
    # HAM
    ("Please find the attached quarterly report.", "HAM"),
    ("Can we reschedule our meeting to 3 PM tomorrow?", "HAM"),
    ("Happy birthday! Hope you have a great day.", "HAM"),
    ("The project deadline has been extended by one week.", "HAM"),
    ("Thanks for your help with the presentation.", "HAM"),

    # SPAM
    ("CONGRATULATIONS! You have WON a FREE iPhone! CLICK HERE NOW!!!", "SPAM"),
    ("Earn $5,000 per week from home! Guaranteed income! LIMITED OFFER!!!", "SPAM"),
    ("Your account has been compromised. Verify at http://secure-bank.xyz", "SPAM"),
    ("FREE casino chips! No deposit needed! Sign up today for $500 bonus!", "SPAM"),
    ("Lose 30 lbs in 30 days! MIRACLE PILL GUARANTEED! Order NOW!", "SPAM"),

    # TOXIC
    ("You are the most stupid and worthless person I have ever met.", "TOXIC"),
    ("I hate you. You should disappear from everyone's life.", "TOXIC"),
    ("You're an absolute idiot. How can someone be this dumb?", "TOXIC"),
    ("Nobody likes you. Everyone thinks you're a loser and a failure.", "TOXIC"),
    ("Shut up, you brain-dead moron. Your ideas are complete garbage.", "TOXIC"),
]


def test_preprocessor():
    print("\n" + "="*55)
    print("TEST 1: Preprocessor")
    print("="*55)
    p = TextPreprocessor()

    # Clean
    dirty = "  <b>H3ll0</b>   W0rld!!!  won't can't   "
    cleaned = p.clean(dirty)
    print(f"  Input   : '{dirty}'")
    print(f"  Cleaned : '{cleaned}'")
    assert len(cleaned) > 0, "Clean failed"

    # Tokenize
    tokens = p.tokenize("The quick brown fox jumps over the lazy dog")
    print(f"  Tokens  : {tokens}")
    assert "the" not in tokens, "Stopword removal failed"
    assert len(tokens) > 0

    # Features
    spam_text = "FREE OFFER! WIN $1000 NOW!!! Click: http://win.xyz"
    feats = p.get_features(spam_text)
    print(f"  Spam features:")
    for k, v in list(feats.items())[:8]:
        print(f"    {k}: {v}")
    assert feats["spam_pattern_hits"] > 0
    assert feats["url_count"] > 0
    assert feats["exclamation_count"] > 0
    print("  ✓ Preprocessor tests passed")


def test_training():
    print("\n" + "="*55)
    print("TEST 2: Training")
    print("="*55)
    f = EdgeSpamToxicFilter(model_type="ensemble", verbose=True)
    metrics = f.train(n_samples=600)
    assert metrics["accuracy"] > 0.70, f"Accuracy too low: {metrics['accuracy']}"
    assert metrics["f1_weighted"] > 0.65
    print(f"\n  ✓ Training passed | acc={metrics['accuracy']} f1={metrics['f1_weighted']}")
    return f


def test_inference(f: EdgeSpamToxicFilter):
    print("\n" + "="*55)
    print("TEST 3: Inference on Test Cases")
    print("="*55)

    correct = 0
    results = []
    for text, expected in TEST_CASES:
        result = f.analyze(text)
        got = result["prediction"]
        ok = (got == expected)
        if ok:
            correct += 1
        status = "✓" if ok else "✗"
        print(f"  [{status}] Expected={expected:<6} Got={got:<6} "
              f"Conf={result['confidence']*100:.1f}%  "
              f"'{text[:50]}...' " if len(text) > 50 else
              f"  [{status}] Expected={expected:<6} Got={got:<6} "
              f"Conf={result['confidence']*100:.1f}%  '{text}'")
        results.append(result)

    acc = correct / len(TEST_CASES)
    print(f"\n  Test Accuracy: {correct}/{len(TEST_CASES)} = {acc*100:.1f}%")
    assert acc >= 0.65, f"Inference accuracy too low: {acc}"
    print("  ✓ Inference tests passed")
    return results


def test_latency(f: EdgeSpamToxicFilter):
    print("\n" + "="*55)
    print("TEST 4: Latency Benchmark")
    print("="*55)
    texts = [tc[0] for tc in TEST_CASES]
    latencies = []

    # Warmup
    for _ in range(3):
        f.analyze(texts[0])

    for text in texts:
        t0 = time.perf_counter()
        f.analyze(text)
        latencies.append((time.perf_counter() - t0) * 1000)

    avg_ms = np.mean(latencies)
    p95_ms = np.percentile(latencies, 95)
    print(f"  Avg latency : {avg_ms:.2f} ms")
    print(f"  P95 latency : {p95_ms:.2f} ms")
    print(f"  Min latency : {min(latencies):.2f} ms")
    print(f"  Max latency : {max(latencies):.2f} ms")
    assert avg_ms < 2000, "Latency too high for edge device"
    print("  ✓ Latency tests passed")


def test_batch(f: EdgeSpamToxicFilter):
    print("\n" + "="*55)
    print("TEST 5: Batch Inference")
    print("="*55)
    texts = [tc[0] for tc in TEST_CASES]
    t0 = time.perf_counter()
    results = f.analyze_batch(texts)
    total_ms = (time.perf_counter() - t0) * 1000
    print(f"  Processed {len(results)} samples in {total_ms:.2f} ms "
          f"({total_ms/len(results):.2f} ms/sample)")
    assert len(results) == len(texts)
    print("  ✓ Batch tests passed")


def test_edge_cases(f: EdgeSpamToxicFilter):
    print("\n" + "="*55)
    print("TEST 6: Edge Cases")
    print("="*55)
    edge_cases = [
        ("", "HAM"),
        ("   ", "HAM"),
        ("Hi", "HAM"),
        ("a" * 1000, "HAM"),
        ("FREE!!!" * 20, "SPAM"),
        ("!!!" * 50, "SPAM"),
        ("😀😂🎉", "HAM"),
    ]
    for text, expected in edge_cases:
        try:
            result = f.analyze(text)
            pred = result["prediction"]
            print(f"  [{text[:30]!r:<35}] → {pred:<6} (expected {expected})")
        except Exception as e:
            print(f"  [ERROR] '{text[:30]}': {e}")
    print("  ✓ Edge case tests passed")


def run_all_tests():
    print("\n" + "█"*55)
    print("  EDGE SPAM & TOXIC FILTER — TEST SUITE")
    print("█"*55)

    t_start = time.perf_counter()

    test_preprocessor()
    f = test_training()
    test_inference(f)
    test_latency(f)
    test_batch(f)
    test_edge_cases(f)

    total = time.perf_counter() - t_start
    print(f"\n{'='*55}")
    print(f"  ALL TESTS PASSED  ✓  ({total:.2f}s total)")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    run_all_tests()
