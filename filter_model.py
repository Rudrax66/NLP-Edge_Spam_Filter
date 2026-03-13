"""
Edge-Device Filter Models
Three complementary classifiers stacked into an ensemble.
All sklearn-based — no GPU, no API, no internet required.
"""

import os
import json
import joblib
import numpy as np
from typing import List, Dict, Tuple, Optional

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score
)
from scipy.sparse import hstack, csr_matrix

LABEL_NAMES = {0: "HAM", 1: "SPAM", 2: "TOXIC"}
MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved")


# ─────────────────────────────────────────────────────────────────────────────
# Feature union: TF-IDF (char + word) + hand-crafted features
# ─────────────────────────────────────────────────────────────────────────────

class FeatureBuilder:
    """Combines TF-IDF text features with hand-crafted numeric features."""

    def __init__(self):
        self.word_tfidf = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 3),
            max_features=15_000,
            min_df=1,
            sublinear_tf=True,
            strip_accents='unicode',
            token_pattern=r'\b[a-zA-Z0-9]\w+\b'
        )
        self.char_tfidf = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 5),
            max_features=10_000,
            min_df=1,
            sublinear_tf=True
        )
        self.scaler = StandardScaler(with_mean=False)
        self.fitted = False

    def fit(self, texts: List[str], handcrafted: np.ndarray) -> 'FeatureBuilder':
        self.word_tfidf.fit(texts)
        self.char_tfidf.fit(texts)
        self.scaler.fit(handcrafted)
        self.fitted = True
        return self

    def transform(self, texts: List[str], handcrafted: np.ndarray):
        word_feat = self.word_tfidf.transform(texts)
        char_feat = self.char_tfidf.transform(texts)
        hc_feat = csr_matrix(self.scaler.transform(handcrafted))
        return hstack([word_feat, char_feat, hc_feat])

    def fit_transform(self, texts: List[str], handcrafted: np.ndarray):
        self.fit(texts, handcrafted)
        return self.transform(texts, handcrafted)


# ─────────────────────────────────────────────────────────────────────────────
# Individual models
# ─────────────────────────────────────────────────────────────────────────────

def build_logistic_regression() -> LogisticRegression:
    return LogisticRegression(
        C=5.0, max_iter=1000, solver='lbfgs',
        class_weight='balanced',
        random_state=42
    )

def build_svm() -> CalibratedClassifierCV:
    return CalibratedClassifierCV(
        LinearSVC(C=1.0, max_iter=2000, class_weight='balanced', random_state=42),
        cv=3
    )

def build_gradient_boost() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42
    )

def build_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_split=2,
        class_weight='balanced', n_jobs=-1, random_state=42
    )

def build_naive_bayes():
    # ComplementNB works better on imbalanced text data
    return ComplementNB(alpha=0.1)


# ─────────────────────────────────────────────────────────────────────────────
# Main EdgeFilter class
# ─────────────────────────────────────────────────────────────────────────────

class EdgeFilter:
    """
    Multi-class spam & toxic filter designed for edge devices.
    Uses a stacked ensemble of lightweight ML models.
    """

    def __init__(self, model_type: str = "ensemble"):
        """
        model_type: 'logistic' | 'svm' | 'naive_bayes' | 'random_forest' |
                    'gradient_boost' | 'ensemble' (default)
        """
        self.model_type = model_type
        self.feature_builder = FeatureBuilder()
        self.model = None
        self.is_trained = False
        self.metadata: Dict = {}

    def _build_model(self):
        if self.model_type == "logistic":
            return build_logistic_regression()
        elif self.model_type == "svm":
            return build_svm()
        elif self.model_type == "naive_bayes":
            return build_naive_bayes()
        elif self.model_type == "random_forest":
            return build_random_forest()
        elif self.model_type == "gradient_boost":
            return build_gradient_boost()
        else:  # ensemble
            # Soft-voting ensemble of diverse models
            return VotingClassifier(
                estimators=[
                    ('lr', build_logistic_regression()),
                    ('svm', build_svm()),
                    ('nb', build_naive_bayes()),
                ],
                voting='soft',
                weights=[3, 2, 1]
            )

    def train(
        self,
        texts: List[str],
        labels: List[int],
        handcrafted_features: np.ndarray,
        test_size: float = 0.2,
        verbose: bool = True
    ) -> Dict:
        """Train the filter and return evaluation metrics."""
        # Split
        (X_tr_txt, X_te_txt,
         X_tr_hc,  X_te_hc,
         y_tr,     y_te) = train_test_split(
            texts, handcrafted_features, labels,
            test_size=test_size, stratify=labels, random_state=42
        )

        # Build features
        if verbose:
            print("[*] Building features...")
        X_tr = self.feature_builder.fit_transform(X_tr_txt, X_tr_hc)
        X_te = self.feature_builder.transform(X_te_txt, X_te_hc)

        # Train
        if verbose:
            print(f"[*] Training {self.model_type} model on {X_tr.shape[0]} samples...")
        self.model = self._build_model()
        self.model.fit(X_tr, y_tr)
        self.is_trained = True

        # Evaluate
        y_pred = self.model.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        f1  = f1_score(y_te, y_pred, average='weighted')
        report = classification_report(
            y_te, y_pred,
            target_names=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)],
            output_dict=True
        )
        cm = confusion_matrix(y_te, y_pred)

        self.metadata = {
            "model_type": self.model_type,
            "train_samples": len(X_tr_txt),
            "test_samples": len(X_te_txt),
            "accuracy": round(acc, 4),
            "f1_weighted": round(f1, 4),
            "report": report,
        }

        if verbose:
            print(f"\n{'='*55}")
            print(f"  Model      : {self.model_type.upper()}")
            print(f"  Accuracy   : {acc*100:.2f}%")
            print(f"  F1 (wtd)   : {f1*100:.2f}%")
            print(f"{'='*55}")
            print("\nClassification Report:")
            print(classification_report(
                y_te, y_pred,
                target_names=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]
            ))
            print("Confusion Matrix (rows=actual, cols=predicted):")
            print(f"       HAM   SPAM  TOXIC")
            for i, row in enumerate(cm):
                print(f"  {LABEL_NAMES[i]:<6} {row}")

        return self.metadata

    def predict(
        self,
        text: str,
        handcrafted: np.ndarray
    ) -> Tuple[int, str, Dict[str, float]]:
        """
        Returns (label_int, label_name, confidence_dict)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        X = self.feature_builder.transform([text], handcrafted.reshape(1, -1))
        label_int = int(self.model.predict(X)[0])

        # Probabilities
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            conf = {LABEL_NAMES[i]: round(float(p), 4) for i, p in enumerate(proba)}
        else:
            conf = {LABEL_NAMES[label_int]: 1.0}

        return label_int, LABEL_NAMES[label_int], conf

    def predict_batch(
        self,
        texts: List[str],
        handcrafted: np.ndarray
    ) -> List[Tuple[int, str, Dict]]:
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        X = self.feature_builder.transform(texts, handcrafted)
        preds = self.model.predict(X)
        results = []
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(X)
            for pred, proba in zip(preds, probas):
                conf = {LABEL_NAMES[i]: round(float(p), 4)
                        for i, p in enumerate(proba)}
                results.append((int(pred), LABEL_NAMES[int(pred)], conf))
        else:
            for pred in preds:
                results.append((int(pred), LABEL_NAMES[int(pred)],
                                {LABEL_NAMES[int(pred)]: 1.0}))
        return results

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> str:
        if not self.is_trained:
            raise RuntimeError("Nothing to save — model not trained.")
        os.makedirs(MODEL_DIR, exist_ok=True)
        path = path or os.path.join(MODEL_DIR, f"filter_{self.model_type}.joblib")
        payload = {
            "model": self.model,
            "feature_builder": self.feature_builder,
            "metadata": self.metadata,
            "model_type": self.model_type,
        }
        joblib.dump(payload, path, compress=3)
        print(f"[✓] Model saved → {path}")
        return path

    @classmethod
    def load(cls, path: str) -> 'EdgeFilter':
        payload = joblib.load(path)
        ef = cls(model_type=payload["model_type"])
        ef.model = payload["model"]
        ef.feature_builder = payload["feature_builder"]
        ef.metadata = payload["metadata"]
        ef.is_trained = True
        print(f"[✓] Model loaded ← {path}")
        return ef

    def cross_validate(
        self,
        texts: List[str],
        labels: List[int],
        handcrafted: np.ndarray,
        cv: int = 5
    ) -> Dict:
        """Run k-fold cross-validation."""
        print(f"[*] Running {cv}-fold cross-validation...")
        X = self.feature_builder.fit_transform(texts, handcrafted)
        model = self._build_model()
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        acc_scores  = cross_val_score(model, X, labels, cv=skf, scoring='accuracy', n_jobs=-1)
        f1_scores   = cross_val_score(model, X, labels, cv=skf, scoring='f1_weighted', n_jobs=-1)

        result = {
            "accuracy_mean":  round(acc_scores.mean(), 4),
            "accuracy_std":   round(acc_scores.std(),  4),
            "f1_mean":        round(f1_scores.mean(),  4),
            "f1_std":         round(f1_scores.std(),   4),
            "fold_accuracies": acc_scores.tolist(),
        }
        print(f"  CV Accuracy : {result['accuracy_mean']*100:.2f}% ± {result['accuracy_std']*100:.2f}%")
        print(f"  CV F1 Score : {result['f1_mean']*100:.2f}% ± {result['f1_std']*100:.2f}%")
        return result
