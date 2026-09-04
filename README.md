# Adaptive and Explainable Machine Learning Framework for Phishing URL Detection

**Muhammad Tabish · Student ID: 77655620**
MSc Dissertation · Leeds Beckett University


## Overview

This project implements a machine learning framework that detects phishing URLs
and explains its decisions. It combines four classifiers, an incremental
(batch) retraining pipeline to address concept drift, and two explainability
methods (SHAP and LIME). A Streamlit dashboard presents each prediction together
with a transparent, feature-level explanation.

**Ethics note:** all analysis is performed on the URL string only. The system
never visits, resolves or contacts any URL. The SSL, DNS and region panels shown
in the dashboard are inferred from the URL string, not from live network lookups.


## Research questions

- **RQ1**: What is the impact of incremental learning on long-term detection?
- **RQ2**: Can explainable outputs enhance understanding of, and trust in, automated detection?
- **RQ3**: Which combination of features and algorithms is most effective?


## Repository contents

| File | Purpose |
|------|---------|
| `phisingurldetection.ipynb` | Full training pipeline: data cleaning, feature extraction, model comparison, SHAP, LIME, and adaptive-learning experiment (run in Google Colab) |
| `features.py` | URL feature extraction (30 features) and the dashboard's analysis panels |
| `app.py` | Streamlit dashboard for real-time URL analysis with SHAP and LIME |
| `requirements.txt` | Python dependencies |

---

## Data and model

The dataset and trained model are not stored in this repository due to size.

- **Dataset:** the publicly available *Malicious URLs* dataset (Kaggle,
  651,191 URLs). It was filtered to the phishing and benign classes,
  supplemented with legitimate domains from the Tranco list, and balanced to a
  120,000-record working set (60,000 phishing / 60,000 legitimate).
- **Model:** `best_model.pkl` is produced by running the notebook. It contains
  the selected Random Forest classifier, the model name, and the feature order.

To run the dashboard, place the generated `best_model.pkl` in a `models/`
folder and the processed `dataset.csv` in a `data/` folder alongside `app.py`.

---

## How to run

### 1. Reproduce the model (Google Colab)
Open `phisingurldetection.ipynb` in Google Colab, upload the dataset when
prompted, and run the cells in order. The notebook cleans the data, extracts
features, trains and compares four models (Logistic Regression, Random Forest,
Gradient Boosting, XGBoost), generates SHAP and LIME explanations, runs the
adaptive-learning experiment, and exports `best_model.pkl` and `dataset.csv`.

### 2. Run the dashboard (local machine)
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open http://localhost:8501, paste a URL, and click **Analyse**.

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
URL string alone. Known limitations including dataset distribution bias and
the simulated (non-timestamped) nature of the drift experiment are documented
in full in the accompanying dissertation.
