# Fake News Detection System

AI-assisted fake-news text classification and credibility-risk assessment using TF-IDF, a Linear SVM, a DistilBERT classifier, and supporting analysis modules.

> **Important limitation:** This is a machine-learning assessment tool, not a definitive fact-checker. Predictions can be wrong. Review evidence and trusted sources before treating a claim as true or false.

## Project files

- `app.py` — Flask web application
- `train_model.py` — trains and evaluates Logistic Regression, Multinomial Naive Bayes, and Linear SVM
- `model.pkl`, `vectorizer.pkl` — saved classifier and TF-IDF vectorizer
- `transformer_benchmark.py` — transformer benchmarking script
- `claim_analysis.py`, `evidence_analysis.py`, `source_analysis.py`, `suspicious_signals.py` — supporting analysis modules
- `decision_engine.py`, `risk_engine.py` — combine model outputs and risk signals
- `templates/index.html` — web interface

## Setup (Python 3.11 recommended)

Create and activate a virtual environment, then install dependencies.

### Windows
```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the application

```bash
python app.py
```

Open `http://127.0.0.1:5050`. On first run, Hugging Face transformer weights and the sentence-transformer model may need to download, so internet access is required. Downloads can take time and disk space.

## Training / dataset

`train_model.py` expects `data/dataset.csv` with these columns:

- `title`
- `text`
- `label` — current script assumes `0 = Fake`, `1 = Real`

The dataset is **not included in this repository ZIP**. Place an appropriately licensed dataset at `data/dataset.csv` before running:

```bash
python train_model.py
```

Training overwrites `model.pkl` and `vectorizer.pkl`; back up the existing files first if you want to preserve them.

## Model artifact compatibility

The included `.pkl` artifacts were serialized with scikit-learn 1.6.1. Use the pinned version in `requirements.txt` to reduce compatibility warnings. Only load pickle/joblib artifacts from a trusted source.

## Current limitations

- The training dataset is not bundled, so retraining is not immediately reproducible.
- The application depends on external pretrained models from Hugging Face.
- URL evidence retrieval depends on the target website being reachable and allowing automated requests.
- Model confidence is not the same as factual certainty; treat retrieved evidence and risk signals as supporting indicators, not proof.
