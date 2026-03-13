# Edge-Device Spam & Toxic Filter
### NLP Project | Python | No API Key Required | Runs 100% Offline

---

## Overview

A full-fledged **multi-class NLP text classifier** designed to run entirely on edge
devices (laptops, Raspberry Pi, embedded systems) without any internet connection,
cloud API, or GPU.

### What it detects

| Label  | Description                              | Action       |
|--------|------------------------------------------|--------------|
| `HAM`  | Clean, legitimate message                | ALLOW        |
| `SPAM` | Unsolicited/malicious promotional text   | BLOCK / FLAG |
| `TOXIC`| Abusive, hateful, or threatening content | BLOCK / FLAG |

---

## Project Structure

```
edge_spam_filter/
├── main.py                  ← Entry point (train + compare + test)
├── pipeline.py              ← End-to-end pipeline (EdgeSpamToxicFilter)
├── demo.py                  ← Interactive CLI demo
│
├── utils/
│   └── preprocessor.py      ← Text cleaning + hand-crafted feature extraction
│
├── models/
│   ├── filter_model.py      ← ML models + FeatureBuilder + EdgeFilter
│   └── saved/               ← Persisted model files (.joblib)
│
├── data/
│   └── dataset.py           ← Synthetic dataset generator (600+ samples)
│
└── tests/
    └── test_filter.py       ← Full test suite (6 test categories)
```

---

## Architecture

```
Input Text
    │
    ▼
┌─────────────────────┐
│   TextPreprocessor   │  Unicode norm → HTML strip → Contraction expand
│   (preprocessor.py) │  → Obfuscation decode → Whitespace normalize
└─────────────────────┘
    │
    ├── Cleaned text ──────────────┐
    └── Hand-crafted features ─────┤
         (22 numeric signals)      │
                                   ▼
                        ┌──────────────────────┐
                        │    FeatureBuilder     │
                        │  Word TF-IDF (15k)    │
                        │  Char TF-IDF (10k)    │
                        │  + Scaled HC features │
                        └──────────────────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │   Ensemble Classifier │
                        │  ┌─────────────────┐ │
                        │  │ LogisticReg (w=3)│ │
                        │  │ LinearSVC   (w=2)│ │
                        │  │ ComplementNB (w=1)│ │
                        │  └─────────────────┘ │
                        │   Soft Voting         │
                        └──────────────────────┘
                                   │
                                   ▼
              ┌────────────────────────────────────┐
              │  {prediction, confidence, action,   │
              │   all_scores, latency_ms, features} │
              └────────────────────────────────────┘
```

---

## Quick Start

```bash
# 1. Install dependencies (only stdlib + scikit-learn + numpy + scipy + joblib)
pip install scikit-learn numpy scipy joblib pandas

# 2. Run the full pipeline
python main.py

# 3. Compare all models
python main.py --compare

# 4. Run test suite
python main.py --test

# 5. Interactive demo (type your own messages)
python demo.py
```

---

## Usage in Code

```python
from pipeline import EdgeSpamToxicFilter

# Create and train
f = EdgeSpamToxicFilter(model_type="ensemble")
f.train(n_samples=600)

# Analyse a single message
result = f.analyze("CONGRATULATIONS! You WON a FREE iPhone! CLICK NOW!!!")
print(result["prediction"])      # SPAM
print(result["confidence"])      # 0.99
print(result["action"])          # BLOCK

# Analyse a batch
results = f.analyze_batch([
    "Can we meet at 3 PM?",
    "FREE MONEY!!! Click here NOW!!!",
    "You're stupid and worthless.",
])

# Save / load model
f.save("my_filter.joblib")
f2 = EdgeSpamToxicFilter()
f2.load("my_filter.joblib")
```

---

## Features Extracted

### Hand-Crafted (22 signals)

| Category   | Features                                                    |
|------------|-------------------------------------------------------------|
| Length     | char_count, word_count, avg_word_len, sentence_count        |
| Lexical    | unique_word_ratio, digit_ratio, upper_ratio, punct_ratio    |
| Spam       | exclamation_count, url_count, dollar_count, spam_pattern_hits|
| Toxic      | toxic_pattern_hits, has_profanity_proxy                     |
| Structural | has_url, has_email, has_phone, repeated_chars, line_count   |

### TF-IDF (vectorized)

- **Word n-grams** (1–3): 15,000 features, sublinear TF scaling
- **Character n-grams** (2–5): 10,000 features, handles obfuscation

---

## Available Models

| Model          | Accuracy | Train Time | Best For           |
|----------------|----------|------------|--------------------|
| `naive_bayes`  | ~100%    | 0.25s      | Ultra-fast/minimal |
| `svm`          | ~100%    | 1.2s       | Fast + robust      |
| `random_forest`| ~100%    | 1.2s       | Feature importance |
| `logistic`     | ~100%    | 27s        | Interpretable      |
| `ensemble`     | ~100%    | 30s        | Best overall       |

---

## Performance

- **Avg inference latency**: ~5 ms per message
- **Batch throughput**: ~3.5 ms per message
- **Memory footprint**: ~50–200 MB (model dependent)
- **No GPU required**
- **No internet required**
- **No API key required**

---

## Confidence Tiers

| Tier   | Threshold | Meaning                           |
|--------|-----------|-----------------------------------|
| HIGH   | ≥ 85%     | Very confident → BLOCK            |
| MEDIUM | ≥ 60%     | Moderately confident → FLAG/BLOCK |
| LOW    | < 60%     | Uncertain → FLAG for human review |

---

## Dependencies

```
scikit-learn >= 1.0
numpy >= 1.20
scipy >= 1.7
joblib >= 1.0
pandas >= 1.3    (optional, for data handling)
```

Standard library only otherwise — no NLTK, no spaCy, no transformers, no API.

---

## Extending the Filter

### Add your own training data

```python
texts = ["Your message here", ...]
labels = [0, 1, 2, ...]  # 0=ham, 1=spam, 2=toxic

f = EdgeSpamToxicFilter()
f.train(texts=texts, labels=labels)
```

### Add a new category

1. Add label in `LABEL_NAMES` in `filter_model.py`
2. Add samples in `dataset.py`
3. Add detection patterns in `preprocessor.py`

---

*Built for the Edge-Device NLP course project — runs anywhere Python runs.*
