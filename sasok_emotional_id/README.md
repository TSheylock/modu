# SASOK Emotional Mirror — Integration Guide

**Module:** `sasok_emotional_id`
**Purpose:** Generate a primary "Emotional ID" profile through controlled OWL dialog sessions, extract 12 emotional features, validate stability across 3 runs, and produce a markdown report with visualizations.

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.10+ | Tested on 3.10 and 3.11 |
| OWL running locally | `http://127.0.0.1:7860/` — select **Function Module = run** |
| `SASOK_AES_KEY_PATH` env var | Path to a 32-byte AES-256 key (see below) |

---

## Installation

```bash
# 1. From the repository root
cd sasok_emotional_id

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Download NLTK VADER lexicon (one-time)
python -c "import nltk; nltk.download('vader_lexicon')"

# 4. Generate AES-256 key (store OUTSIDE the repo)
python -c "from encrypt import generate_key; generate_key('/secure/path/sasok.key')"
# Then set:
export SASOK_AES_KEY_PATH=/secure/path/sasok.key
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `SASOK_AES_KEY_PATH` | **Yes** | Absolute path to 32-byte AES key file |
| `OWL_ENDPOINT` | No | OWL API endpoint (default: `http://127.0.0.1:7860/run`) |
| `RUN_INTERVAL_SECONDS` | No | Seconds between runs (default: 630 = 10.5 min) |

Create a `.env` file in `sasok_emotional_id/` (never commit it):

```
SASOK_AES_KEY_PATH=/secure/path/sasok.key
OWL_ENDPOINT=http://127.0.0.1:7860/run
```

---

## Running

### Full 3-run session (production)

```bash
python runner.py
```

Sends 12 prompts × 3 runs to OWL, waits ~10.5 min between runs, encrypts responses, extracts 12 features, and writes:

- `outputs/sasok_emotional_id/<session_id>_run<N>_q<M>.json` — per-question JSON
- `outputs/sasok_emotional_id/aggregate.csv` — cumulative CSV

### Dry-run (no OWL, mock responses)

```bash
python runner.py --dry-run
```

### Seed CSV (Test A — 50 synthetic prompts)

```bash
python runner.py --seed-csv seeds.csv --dry-run
```

---

## Validation

```bash
# Run all 3 acceptance tests
python validate_acceptance.py --all --seed-csv seeds.csv

# Individual tests
python validate_acceptance.py --test-a --seed-csv seeds.csv
python validate_acceptance.py --test-b --session-id <uuid>
python validate_acceptance.py --test-c
```

| Test | What it checks |
|------|----------------|
| A — Functional | ≥ 95% parse success on 50 seed prompts |
| B — Stability | CI ≥ 0.6 for ≥ 8 / 11 features |
| C — Noise | Short inputs flagged as `noisy=True` |

---

## Report Generation

```bash
python generate_report.py --session-id <uuid>
```

Writes `reports/<session_id>_report.md` with:

- Executive summary table
- Valence distribution histogram
- Trust indicators bar chart
- Consistency Index heatmap
- 3 recommendations for SASOK model training
- List of high-noise features

---

## Output Artifacts

```
sasok_emotional_id/
├── outputs/sasok_emotional_id/
│   ├── <session_id>_run1_q1.json   ← per-question encrypted record
│   ├── …
│   └── aggregate.csv               ← all sessions / runs / questions
├── logs/
│   └── sasok_emotional_id.log
├── quarantine/                     ← records moved here if >10% fail
└── reports/
    ├── <session_id>_report.md
    ├── <session_id>_valence_dist.png
    ├── <session_id>_trust_bar.png
    └── <session_id>_ci_heatmap.png
```

---

## 12 Emotional Profile Features

| # | Feature | Range | Description |
|---|---------|-------|-------------|
| 1 | `latency_to_emotion` | int | Sentence index of first emotional sentence |
| 2 | `valence` | −1 … +1 | VADER compound sentiment score |
| 3 | `arousal` | 0 … 1 | Avg sentence length proxy + exclamation density |
| 4 | `cognitive_complexity` | 0 … 1 | Sentence length + subordinate conjunctions |
| 5 | `self_reference_rate` | 0 … 1 | Fraction of 1st-person pronouns |
| 6 | `uncertainty_marker_freq` | 0 … 1 | Fraction of uncertainty-marker tokens |
| 7 | `goal_orientedness` | 0 / 1 | Goal-related keywords present |
| 8 | `social_orientation` | 0 … 1 | Fraction of social/collective tokens |
| 9 | `emotional_granularity` | 0 … 1 | Normalised distinct emotion-bearing lexemes |
| 10 | `adaptive_language` | 0 / 1 | Corrective/solution verbs present |
| 11 | `trust_indicators` | −1 … +1 | (pos trust − neg trust) / total tokens |
| 12 | `consistency_index` | 0 … 1 | CI across 3 runs (filled after all runs) |

---

## Security & Privacy

- Raw text is AES-256-GCM encrypted before storage; key lives outside the repo.
- PII (names, emails, phone numbers) is anonymised before export.
- Raw text is retained for a maximum of 30 days (schedule deletion with a cron job).
- Each file access is logged with timestamp and hash for auditability.
- `.env` and key files are listed in `.gitignore`.

---

## Consistency Index Formula

For each feature *p* across 3 runs with values *v₁, v₂, v₃*:

```
μ  = (v₁ + v₂ + v₃) / 3
σ  = std(v₁, v₂, v₃)
CI = 1 − σ / (|μ| + ε)       clamped to [0, 1]
```

When |μ| ≈ 0, the range-based fallback is used:
`CI = 1 − σ / (range + ε)`

Profile is **stable** when CI ≥ 0.6 for at least 8 of 11 features.
