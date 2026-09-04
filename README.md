# Adaptive and Explainable Machine Learning Framework for Phishing URL Detection

**Muhammad Tabish · Student ID: 77655620**
MSc Dissertation · Leeds Beckett University

---

## Overview

This project implements a machine learning framework that detects phishing URLs
and explains its decisions. It combines four classifiers, an incremental
(batch) retraining pipeline to address concept drift, and two explainability
methods (SHAP and LIME). A Streamlit dashboard presents each prediction together
with a transparent, feature-level explanation.

**Ethics note:** all analysis is performed on the URL string only. The system
never visits, resolves or contacts any URL. The SSL, DNS and region panels shown
in the dashboard are inferred from the URL string, not from live network lookups.

---

## Research questions

- **RQ1** — What is the impact of incremental learning on long-term detection?
- **RQ2** — Can explainable outputs enhance understanding of, and trust in, automated detection?
- **RQ3** — Which combination of features and algorithms is most effective?

---

## Repository contents

| File | Purpose |
|------|---------|
| `phishing_url_detection.ipynb` | Full training pipeline: data cleaning, feature extraction, model comparison, SHAP, LIME, and adaptive-learning experiment (run in Google Colab) |
| `features.py` | URL feature extraction (30 features) and the dashboard's analysis panels |
| `app.py` | Streamlit dashboard for real-time URL analysis with SHAP and LIME |
| `dataset.csv` | Processed, balanced dataset (120,000 records) used by the dashboard for LIME |
| `requirements.txt` | Python dependencies |

---

## Trained model (download separately)

The trained model file `best_model.pkl` (46 MB) is too large for this
repository and is hosted on Google Drive:

**Download `best_model.pkl`:** https://drive.google.com/file/d/1RkDFYMNa2SwSlDpaprj-oMkP2Bi7D743/view?usp=sharing

Place it in the same folder as `app.py` (see "How to run" below).

The model was produced by running the notebook. It contains the selected
Random Forest classifier, the model name, and the feature order.

---

## Data

The dataset originates from the publicly available *Malicious URLs* dataset
(Kaggle, 651,191 URLs). It was filtered to the phishing and benign classes,
supplemented with legitimate domains from the Tranco list, and balanced to a
120,000-record working set (60,000 phishing / 60,000 legitimate). The processed
version (`dataset.csv`) is included in this repository.

---

## How to run

### 1. Download the files
- Clone or download this repository (includes the code and `dataset.csv`).
- Download `best_model.pkl` from the Google Drive link above.

### 2. Place all files in one folder
Put everything together in a single folder:
```
project/
├── app.py
├── features.py
├── requirements.txt
├── dataset.csv
└── best_model.pkl   (downloaded from Google Drive)
```

### 3. Install dependencies and run
```
pip install -r requirements.txt
streamlit run app.py
```
Then open http://localhost:8501, paste a URL, and click **Analyse**.

### Reproducing the model (optional)
Open `phishing_url_detection.ipynb` in Google Colab, upload the dataset when
prompted, and run the cells in order. The notebook cleans the data, extracts
features, trains and compares four models, generates SHAP and LIME explanations,
runs the adaptive-learning experiment, and exports `best_model.pkl` and
`dataset.csv`.

---

## Methods summary

- **Features (30):** lexical, domain-based and structural, all string-derived.
- **Models:** Logistic Regression (baseline), Random Forest, Gradient Boosting, XGBoost.
- **Selection:** Random Forest, chosen by highest F1-score on an 80/20 stratified split.
- **Explainability:** SHAP (global and local) and LIME (local).
- **Adaptive learning:** batch retraining to simulate performance under evolving data.

---

## Limitations

Reported performance reflects a balanced test set and features derived from the
URL string alone. Known limitations — including dataset distribution bias and
the simulated (non-timestamped) nature of the drift experiment — are documented
in full in the accompanying dissertation.
