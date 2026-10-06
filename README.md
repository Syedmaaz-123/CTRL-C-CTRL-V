# URLSentinel AI

**Intelligent Phishing URL Detection & Risk Analysis**
**NexGuard AI · v1.0**

URLSentinel AI is a branded Streamlit interface for analyzing URL strings with
the pretrained phishing classifier supplied by the original project. The
NexGuard AI team did not train or alter that model. The application does not
visit URLs, resolve domains, or make network requests to submitted hosts.

> **Original project and model attribution:** Muhammad Tabish (Student ID
> 77655620), MSc dissertation, Leeds Beckett University. This interface reuses the project's supplied
> pretrained artifact and feature extraction code. The source repository has no
> `LICENSE` file; retain this attribution when redistributing the project and
> confirm any applicable permissions before publication.

## Purpose

The application accepts a URL as text, extracts URL structure and lexical
indicators, and passes those values to the existing pretrained Random Forest
model. It displays the model's classification and probability alongside a
separate rule-based threat score and optional SHAP/LIME explanations. It is a
screening aid, not a guarantee that a URL is safe or malicious.

## Architecture

```text
URL entered in app.py
        │
        ├── features.py: normalize string and derive 30 numeric features
        │       └── rule-based URL indicators and threat score
        │
        └── app.py: order features with artifact['feature_cols']
                └── best_model.pkl: predict() and predict_proba()
                        └── result, probability, SHAP/LIME explanations

dataset.csv ── feature reference rows for LIME explanations
```

### URL scanning and prediction

1. The user pastes a URL and selects **Scan URL**.
2. The app treats it as text and calls the existing `full_analysis()` and
   `extract_features()` functions from `features.py`.
3. It creates a one-row DataFrame and selects columns in the exact order stored
   in `best_model.pkl` under `feature_cols`.
4. The saved estimator receives that row through its existing `predict()` and
   `predict_proba()` methods. Label `1` is phishing; label `0` is legitimate.
5. The application shows `PHISHING` / `HIGH` or `LEGITIMATE` / `LOW` according
   to the model class. Confidence is the probability for the predicted class;
   phishing probability is also shown separately.
6. A separate rule-based score and URL indicators summarize patterns in the
   string. They do not modify or override the classifier's prediction.
7. SHAP explains feature contributions. LIME uses the bundled `dataset.csv` as
   reference feature data. If an explanation cannot be computed, the result
   remains visible and the app reports the explanation issue.

The SSL, DNS, and region information available in `features.py` is inferred
from URL text. The application does not verify TLS certificates, query DNS, or
locate a server. HTTPS in a string does not establish that a site is
trustworthy.

## Feature extraction

`features.py` normalizes the URL text (trims whitespace, lowercases it, removes
a leading HTTP/HTTPS scheme and leading `www.`) and derives 30 numeric values.
They cover URL/domain/path lengths, punctuation counts, digits, subdomains,
character entropy, IP-like hosts, ports, shortener strings, suspicious words,
path depth, TLD lists, brand terms, and domain length. These definitions and
their ordering are left as supplied because the saved model expects them.

## Pretrained model

Place the existing model at:

```text
project-root/
├── app.py
├── features.py
├── dataset.csv
└── best_model.pkl
```

The filename must be `best_model.pkl`. `app.py` loads it with `pickle` and
expects a dictionary containing `model`, `model_name`, and `feature_cols`.
This application uses the supplied artifact for inference only. It does not
train, replace, or modify the model. Pickle files can execute code when loaded;
only load an artifact from a trusted source.

The artifact was saved with scikit-learn **1.6.1**. `requirements.txt` lists a
minimum version rather than pinning that version, so a newer installation can
emit a cross-version compatibility warning. For the closest runtime match,
install the listed requirements and then pin the existing dependency:

```bash
python -m pip install "scikit-learn==1.6.1"
```

This changes only the virtual environment; it does not retrain or rewrite the
model.

## Dataset

The included `dataset.csv` contains 120,000 rows: 60,000 label `0` and 60,000
label `1`. Each row has the URL, the 30 extracted feature values, and its label.
The original notebook describes filtering phishing and benign records from
`malicious_phish.csv`, adding benign domains from Tranco, and sampling a
balanced set. The raw `malicious_phish.csv` input is not included. The running
app reads `dataset.csv` as reference data for LIME; it does not train on it.

The original notebook `phisingurldetection.ipynb` contains the training and
evaluation workflow. It compared Logistic Regression, Random Forest, Gradient
Boosting, and XGBoost, then selected the highest-F1 model. The saved notebook
output names Random Forest as the winner. The notebook is included for
provenance; running it is not part of normal app use and is not required to use
the pretrained model.

## Installation

From the project root on macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install "scikit-learn==1.6.1"
```

The last command aligns scikit-learn with the version recorded in the supplied
artifact. To activate an already-created environment later:

```bash
source .venv/bin/activate
```

## Run

From the project root with the environment active and `best_model.pkl` beside
`app.py`:

```bash
streamlit run app.py
```

Open the local address printed by Streamlit (typically
`http://localhost:8501`). Paste a URL and choose **Scan URL**.

## Example scans

Try a familiar legitimate example:

```text
https://www.google.com/
```

Try this intentionally suspicious example from the app's original placeholder:

```text
http://secure-login-paypal.xyz/verify@account
```

The displayed status and confidence come from the loaded model's actual output.
Predictions can vary by input; these examples are not hard-coded expected
answers or guarantees.

## Technology stack

- Python
- Streamlit
- pandas and NumPy
- scikit-learn Random Forest (supplied pretrained artifact)
- SHAP and LIME for explanations
- matplotlib for explanation charts
- XGBoost is listed in the original requirements and training notebook

## Configuration and branding

The top of `app.py` contains the `APP_NAME`, `TAGLINE`, `TEAM_NAME`, `VERSION`,
`LOGO_PATH`, and `COLORS` settings. A local logo can be enabled by setting
`LOGO_PATH` to its file path; a shield icon is used as the current placeholder.

## Limitations and safety

- The model evaluates string-derived features only; the app never opens the URL
  or contacts its host.
- The model can produce false positives and false negatives. A probability is
  the estimator's output, not a guarantee or necessarily a calibrated real
  world likelihood.
- Dataset results came from a balanced test set and may not represent current
  traffic or deployment performance.
- Fixed keyword, brand, shortener, and TLD lists can become outdated and can
  flag legitimate URLs.
- The SSL, DNS, and region summaries are string-based indicators, not live
  security checks.
- The bundled estimator can warn when loaded with a scikit-learn version other
  than the version that saved it.
- The app preserves the supplied model and feature order. Any future model or
  feature changes require a separate compatibility review.

## Project files

| File | Purpose |
|---|---|
| `app.py` | Branded Streamlit dashboard, artifact loading, inference, and explanations |
| `features.py` | Original feature extraction and string-based analysis functions |
| `best_model.pkl` | Pretrained model artifact; supply it in the project root |
| `dataset.csv` | Processed feature data used by LIME |
| `phisingurldetection.ipynb` | Original training, evaluation, and export notebook |
| `requirements.txt` | Python dependencies |

## Attribution

Original project author: **Muhammad Tabish (Student ID 77655620)**, MSc
dissertation, Leeds Beckett University. The supplied model and feature definitions are reused as provided;
NexGuard AI is responsible for this branded application interface and does not
claim to have trained the original model. No `LICENSE` file was present in the
source repository at the time of this update.
