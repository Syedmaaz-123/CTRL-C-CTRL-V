"""URLSentinel AI: local, feature-based phishing URL analysis dashboard."""

import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from lime.lime_tabular import LimeTabularExplainer

from features import extract_features, full_analysis


# Keep all customer-facing brand values in one place for easy white-labeling.
APP_NAME = "URLSentinel AI"
TAGLINE = "Intelligent Phishing URL Detection & Risk Analysis"
TEAM_NAME = "NexGuard AI"
VERSION = "v1.0"
LOGO_PATH = None  # Set to a local image path when a project logo is available.
COLORS = {
    "accent": "#24d3b5",
    "accent_dark": "#0ca58e",
    "on_accent": "#041411",
    "background": "#07111f",
    "surface": "#101d2d",
    "surface_alt": "#142438",
    "text": "#edf4fb",
    "muted": "#9aabc0",
    "danger": "#ff647c",
    "warning": "#ffbd66",
    "success": "#5de0a5",
    "border": "#24384e",
}

BASE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE, "best_model.pkl")
DATA_PATH = os.path.join(BASE, "dataset.csv")

st.set_page_config(
    page_title=APP_NAME,
    page_icon=":material/shield_lock:",
    layout="wide",
)


def apply_brand_styles():
    """Apply a small set of brand colors to the app's main surfaces and CTA."""
    st.markdown(
        f"""
        <style>
        :root {{
            --us-accent: {COLORS['accent']};
            --us-bg: {COLORS['background']};
            --us-surface: {COLORS['surface']};
            --us-text: {COLORS['text']};
            --us-muted: {COLORS['muted']};
            --us-border: {COLORS['border']};
        }}
        [data-testid="stAppViewContainer"] {{ background: var(--us-bg); }}
        [data-testid="stHeader"] {{ background: rgba(7, 17, 31, 0.92); }}
        [data-testid="stMetric"] {{
            background: var(--us-surface);
            border: 1px solid var(--us-border);
            border-radius: 12px;
            padding: 14px 16px;
        }}
        div.stButton > button[kind="primary"],
        div.stFormSubmitButton > button[kind="primary"] {{
            background: {COLORS['accent_dark']};
            border: 1px solid {COLORS['accent']};
            color: {COLORS['on_accent']};
            font-weight: 700;
        }}
        div.stButton > button[kind="primary"]:hover,
        div.stFormSubmitButton > button[kind="primary"]:hover {{
            background: {COLORS['accent']};
            border-color: {COLORS['accent']};
            color: {COLORS['on_accent']};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model():
    """Load the existing pretrained artifact without modifying or retraining it."""
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as model_file:
        return pickle.load(model_file)


@st.cache_data
def load_train(cols):
    """Load feature columns used as the reference data for LIME explanations."""
    if not os.path.exists(DATA_PATH):
        return None
    try:
        return pd.read_csv(DATA_PATH)[cols].values
    except Exception:
        return None


def shap_fig(model, X1, cols):
    """Build a local SHAP feature-contribution chart for the phishing class."""
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X1, check_additivity=False)
    if isinstance(values, list):
        values = values[1]
    values = np.array(values)
    if values.ndim == 3:  # Newer SHAP versions return samples/features/classes.
        values = values[:, :, 1]

    row_values = np.array(values[0]).ravel()[: len(cols)]
    indices = np.argsort(np.abs(row_values))[-10:]
    colors = [COLORS["danger"] if row_values[i] > 0 else COLORS["accent"] for i in indices]

    fig, ax = plt.subplots(figsize=(7, 4), facecolor=COLORS["surface"])
    ax.set_facecolor(COLORS["surface"])
    ax.barh([cols[i] for i in indices], row_values[indices], color=colors)
    ax.axvline(0, color=COLORS["muted"], lw=0.8, ls="--")
    ax.set_xlabel("← toward legitimate     SHAP contribution     toward phishing →", color=COLORS["text"])
    ax.set_title("Feature contributions for this URL", color=COLORS["text"])
    ax.tick_params(colors=COLORS["muted"])
    for spine in ax.spines.values():
        spine.set_color(COLORS["border"])
    fig.tight_layout()
    return fig


def lime_fig(model, X1, cols, Xtrain):
    """Build a local LIME explanation using the bundled feature dataset."""
    explainer = LimeTabularExplainer(
        Xtrain,
        feature_names=cols,
        class_names=["Legitimate", "Phishing"],
        mode="classification",
        discretize_continuous=True,
    )
    explanation = explainer.explain_instance(
        X1.values[0], model.predict_proba, num_features=8
    )
    fig = explanation.as_pyplot_figure()
    fig.set_size_inches(7, 4)
    fig.tight_layout()
    return fig


def show_brand_header(artifact):
    """Render the product identity and the original model attribution."""
    if LOGO_PATH and os.path.isfile(LOGO_PATH):
        st.logo(LOGO_PATH, size="large", icon_image=":material/shield_lock:")

    model_name = artifact.get("model_name", "Pretrained model") if artifact else "Model unavailable"
    with st.container(border=True):
        st.title(APP_NAME, icon=":material/shield_lock:")
        st.write(f"**{TAGLINE}**")
        st.caption(f"{TEAM_NAME} · {VERSION} · Model: {model_name}")


def show_model_information(artifact):
    """Keep model provenance and runtime status visible in the sidebar."""
    with st.sidebar:
        st.subheader("Model information")
        if artifact:
            st.success(f"Loaded: {artifact.get('model_name', 'Pretrained model')}")
            st.caption(f"Input features: {len(artifact['feature_cols'])}")
            st.caption("Inference only · no retraining")
        else:
            st.error("Pretrained model not found")
            st.caption("Place best_model.pkl beside app.py to enable ML predictions.")
        st.caption(f"Application version {VERSION}")
        st.divider()
        st.caption(
            "Original research and model attribution: Muhammad Tabish (Student ID "
            "77655620), MSc dissertation, Leeds Beckett University. URLSentinel AI is "
            "a branded interface and does not claim to have trained the supplied model."
        )


def show_url_indicators(analysis):
    """Show indicators that features.py actually derives from the URL string."""
    features = analysis["features"]
    ssl = analysis["ssl"]
    tokens = analysis["tokens"]

    metric_columns = st.columns(4)
    metric_columns[0].metric("URL length", features["url_length"])
    metric_columns[1].metric("Dots", features["num_dots"])
    metric_columns[2].metric("Subdomains", features["num_subdomains"])
    metric_columns[3].metric("Digits", features["num_digits"])

    indicators = pd.DataFrame(
        [
            ("IP-like host", "Detected" if features["has_ip"] else "Not detected"),
            ("URL scheme", ssl["scheme"].upper()),
            ("Suspicious keywords", ", ".join(tokens["suspicious_tokens"]) or "None found"),
            ("Brand terms", ", ".join(tokens["brand_terms"]) or "None found"),
            (
                "Special characters",
                f"hyphens {features['num_hyphens']} · @ {features['num_at_signs']} · "
                f"? {features['num_question_marks']} · % {features['num_percent']}",
            ),
            ("Top-level domain", analysis["dns"]["tld"]),
        ],
        columns=["Indicator", "Observed value"],
    )
    st.dataframe(indicators, hide_index=True, width="stretch")
    st.caption(
        "Indicators are derived from the URL text. HTTPS shows only the written scheme; "
        "the app does not verify certificates, resolve DNS, or contact the host."
    )


def show_explanations(artifact, model, X1, cols, analysis):
    """Separate URL heuristics from model-specific SHAP and LIME explanations."""
    st.subheader("Why this result")
    st.markdown("**URL indicators (separate from the model decision)**")
    reasons = analysis["threat"]["reasons"]
    if reasons:
        for points, reason in reasons:
            st.write(f"+{points} points · {reason}")
    else:
        st.write("No configured rule-based risk indicators were found.")

    if not artifact or model is None or X1 is None:
        st.info("Model-based explanations are unavailable until best_model.pkl is present.")
        return

    st.markdown("**Model explanations**")
    shap_column, lime_column = st.columns(2)
    with shap_column:
        st.markdown("SHAP")
        try:
            st.pyplot(shap_fig(model, X1, cols), clear_figure=True)
        except Exception as exc:
            st.warning(f"SHAP explanation could not be rendered: {exc}")
    with lime_column:
        st.markdown("LIME")
        reference_data = load_train(cols)
        if reference_data is None:
            st.info("LIME reference features are unavailable in dataset.csv.")
        else:
            try:
                st.pyplot(lime_fig(model, X1, cols, reference_data), clear_figure=True)
            except Exception as exc:
                st.warning(f"LIME explanation could not be rendered: {exc}")


def main():
    apply_brand_styles()
    artifact = load_model()
    show_brand_header(artifact)
    show_model_information(artifact)

    st.header("URL scanner", icon=":material/search:")
    st.write("Analyze URL text locally. Submitted URLs are never opened or contacted.")
    with st.form("url_scan_form", border=True):
        url = st.text_input(
            "URL to scan",
            placeholder="https://example.com/account/login",
            help="Paste the URL as text; the scanner will not visit it.",
        )
        submitted = st.form_submit_button(
            "Scan URL", type="primary", icon=":material/radar:"
        )

    if not submitted:
        st.info("Enter a URL above and select Scan URL to see its analysis.")
        return
    if not url.strip():
        st.warning("Enter a URL before scanning.")
        return

    # The URL is treated strictly as text. No request is made to its host.
    url = url.strip()
    analysis = full_analysis(url)
    st.subheader("URL analysis")
    st.code(url, language=None)

    model = None
    X1 = None
    columns = None
    predicted_phishing = None
    probabilities = None
    if artifact:
        model, columns = artifact["model"], artifact["feature_cols"]
        # Preserve the model's saved feature order and existing inference calls.
        X1 = pd.DataFrame([extract_features(url)])[columns]
        probabilities = model.predict_proba(X1)[0]
        predicted_phishing = model.predict(X1)[0] == 1

    st.subheader("Detection result")
    if predicted_phishing is None:
        st.error("ML prediction unavailable because best_model.pkl was not found.")
    else:
        status = "PHISHING" if predicted_phishing else "LEGITIMATE"
        risk = "HIGH" if predicted_phishing else "LOW"
        confidence = probabilities[1] if predicted_phishing else probabilities[0]
        result_columns = st.columns(3)
        result_columns[0].metric("Status", status)
        result_columns[1].metric("Risk level", risk)
        result_columns[2].metric("Model confidence", f"{confidence * 100:.1f}%")

        st.progress(
            float(probabilities[1]),
            text=f"Model phishing probability: {probabilities[1] * 100:.1f}%",
        )

    st.subheader("Risk assessment")
    risk_columns = st.columns(2)
    risk_columns[0].metric("Rule-based threat score", f"{analysis['threat']['score']} / 100")
    risk_columns[1].metric("Rule-based indicator level", analysis["threat"]["level"])
    st.caption(
        "The threat score is a separate fixed-rule summary; it does not change the ML prediction."
    )

    st.subheader("Security indicators")
    show_url_indicators(analysis)

    st.subheader("Feature analysis")
    feature_table = pd.DataFrame(
        analysis["features"].items(), columns=["Feature", "Value"]
    )
    st.dataframe(feature_table, hide_index=True, width="stretch")

    show_explanations(artifact, model, X1, columns, analysis)

    st.caption(
        f"{APP_NAME} {VERSION} · {TEAM_NAME} · Original model attribution: "
        "Muhammad Tabish (Student ID 77655620). "
        "String-based analysis only; URLs are not visited."
    )


if __name__ == "__main__":
    main()
