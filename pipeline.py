"""
Edge-Device Spam & Toxic Filter — Main Pipeline
Ties together preprocessing → feature extraction → model → inference.
"""

import os
import sys
import time
import numpy as np
from typing import List, Dict, Tuple, Optional

# Local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessor import TextPreprocessor
from models.filter_model import EdgeFilter, LABEL_NAMES
from data.dataset import generate_dataset

# ─────────────────────────────────────────────────────────────────────────────
# Risk thresholds for confidence-based escalation
# ─────────────────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLDS = {
    "HIGH":   0.85,
    "MEDIUM": 0.60,
    "LOW":    0.00,
}


def confidence_tier(prob: float) -> str:
    if prob >= CONFIDENCE_THRESHOLDS["HIGH"]:
        return "HIGH"
    if prob >= CONFIDENCE_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "LOW"


class EdgeSpamToxicFilter:
    """
    End-to-end edge-device filter.
    Usage:
        f = EdgeSpamToxicFilter()
        f.train()
        result = f.analyze("You won a free iPhone!!!")
    """

    def __init__(self, model_type: str = "ensemble", verbose: bool = True):
        self.preprocessor = TextPreprocessor()
        self.model = EdgeFilter(model_type=model_type)
        self.verbose = verbose
        self._trained = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        texts: Optional[List[str]] = None,
        labels: Optional[List[int]] = None,
        n_samples: int = 600,
        cross_validate: bool = False
    ) -> Dict:
        """
        Train from provided data or auto-generate a synthetic dataset.
        labels: 0=ham, 1=spam, 2=toxic
        """
        if texts is None or labels is None:
            if self.verbose:
                print("[*] Generating synthetic training dataset...")
            texts, labels = generate_dataset(
                n_ham=n_samples // 3,
                n_spam=n_samples // 3,
                n_toxic=n_samples // 3,
            )

        if self.verbose:
            counts = {k: labels.count(k) for k in [0, 1, 2]}
            print(f"[*] Dataset: {len(texts)} samples | "
                  f"Ham={counts[0]} Spam={counts[1]} Toxic={counts[2]}")

        # Extract hand-crafted features
        hc_features = self._extract_hc_features(texts)

        if cross_validate:
            cv_results = self.model.cross_validate(texts, labels, hc_features)
            print("\n[Cross-Validation Results]")
            for k, v in cv_results.items():
                print(f"  {k}: {v}")

        metrics = self.model.train(texts, labels, hc_features, verbose=self.verbose)
        self._trained = True
        return metrics

    # ── Inference ─────────────────────────────────────────────────────────────

    def analyze(self, text: str) -> Dict:
        """
        Analyze a single text. Returns rich result dict.
        """
        if not self._trained:
            raise RuntimeError("Filter not trained. Call train() first.")

        t0 = time.perf_counter()

        # Preprocess
        cleaned = self.preprocessor.clean(text)
        features_dict = self.preprocessor.get_features(text)
        hc_vec = np.array(self.preprocessor.features_to_vector(features_dict))

        # Predict
        label_int, label_name, confidences = self.model.predict(cleaned, hc_vec)

        top_conf = confidences[label_name]
        tier = confidence_tier(top_conf)
        latency_ms = (time.perf_counter() - t0) * 1000

        result = {
            "original_text":  text,
            "cleaned_text":   cleaned,
            "prediction":     label_name,
            "label":          label_int,
            "confidence":     top_conf,
            "confidence_tier": tier,
            "all_scores":     confidences,
            "latency_ms":     round(latency_ms, 3),
            "features": {
                "spam_pattern_hits":  features_dict["spam_pattern_hits"],
                "toxic_pattern_hits": features_dict["toxic_pattern_hits"],
                "url_count":          features_dict["url_count"],
                "exclamation_count":  features_dict["exclamation_count"],
                "all_caps_words":     features_dict["all_caps_word_count"],
                "digit_ratio":        round(features_dict["digit_ratio"], 3),
                "upper_ratio":        round(features_dict["upper_ratio"], 3),
            },
            "action": self._recommended_action(label_name, tier),
        }
        return result

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Batch analyze a list of texts."""
        return [self.analyze(t) for t in texts]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> str:
        return self.model.save(path)

    def load(self, path: str) -> None:
        self.model = EdgeFilter.load(path)
        self._trained = True

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_hc_features(self, texts: List[str]) -> np.ndarray:
        rows = []
        for t in texts:
            fd = self.preprocessor.get_features(t)
            rows.append(self.preprocessor.features_to_vector(fd))
        return np.array(rows, dtype=np.float32)

    @staticmethod
    def _recommended_action(label: str, tier: str) -> str:
        if label == "HAM":
            return "ALLOW"
        elif label == "SPAM":
            return "BLOCK" if tier in ("HIGH", "MEDIUM") else "FLAG_REVIEW"
        else:  # TOXIC
            return "BLOCK" if tier == "HIGH" else "FLAG_REVIEW"

    def print_result(self, result: Dict) -> None:
        """Pretty-print analysis result."""
        icons = {"HAM": "✅", "SPAM": "🚫", "TOXIC": "⚠️"}
        colors = {
            "HAM": "\033[92m",   # green
            "SPAM": "\033[91m",  # red
            "TOXIC": "\033[93m", # yellow
        }
        RESET = "\033[0m"
        BOLD  = "\033[1m"

        pred = result["prediction"]
        color = colors.get(pred, "")

        print(f"\n{'─'*60}")
        print(f"  Text    : {result['original_text'][:80]}")
        print(f"  {BOLD}Result  : {color}{icons[pred]} {pred} "
              f"({result['confidence']*100:.1f}% {result['confidence_tier']}){RESET}")
        print(f"  Action  : {BOLD}{result['action']}{RESET}")
        print(f"  Scores  : " + " | ".join(
            f"{k}={v*100:.1f}%" for k, v in result["all_scores"].items()
        ))
        print(f"  Signals : spam_hits={result['features']['spam_pattern_hits']}  "
              f"toxic_hits={result['features']['toxic_pattern_hits']}  "
              f"urls={result['features']['url_count']}  "
              f"caps={result['features']['all_caps_words']}")
        print(f"  Latency : {result['latency_ms']} ms")
        print(f"{'─'*60}")
