"""
app.py — Phishing URL Detector (SOC card-style dashboard)
--------------------------------------------------------
Card-based security-console layout focused on the CURRENT URL result.
Shows: ML verdict + confidence, threat score, SSL/DNS/region/token cards,
and SHAP + LIME explanations side by side.

Run:  streamlit run app.py

ETHICS NOTE: SSL / DNS / region panels are derived from the URL STRING only.
The app never visits or contacts the URL.
"""

import os, pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer

from features import extract_features, full_analysis

BASE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE, "models", "best_model.pkl")
DATA_PATH  = os.path.join(BASE, "data", "dataset.csv")

st.set_page_config(page_title="Phishing URL Detector",
                   page_icon="🛡️", layout="wide")

st.markdown("""
<style>
  .card {background:#ffffff; border:1px solid #e6e8eb; border-radius:12px;
         padding:1rem 1.2rem; margin-bottom:0.8rem;}
  .stat {background:#f6f8fa; border-radius:10px; padding:0.9rem 1rem;}
  .stat .lbl {font-size:0.8rem; color:#6b7280;}
  .stat .val {font-size:1.6rem; font-weight:600;}
  .pill {display:inline-block; padding:3px 12px; border-radius:20px;
         font-size:0.8rem; font-weight:600;}
  .p-red{background:#fdecec; color:#c0392b;} .p-amber{background:#fef6e7; color:#b9770e;}
  .p-green{background:#eafaf1; color:#1e8449;} .p-grey{background:#eef0f2; color:#566573;}
  .mono{font-family:ui-monospace,Menlo,monospace; font-size:0.85rem; word-break:break-all;}
  .topbar{display:flex; align-items:center; justify-content:space-between;
          background:#ffffff; border:1px solid #e6e8eb; border-radius:12px;
          padding:0.7rem 1.2rem; margin-bottom:1rem;}
  .topbar .brand{font-size:1.15rem; font-weight:600; color:#1f2937;}
  .topbar .brand small{font-size:0.72rem; font-weight:400; color:#6b7280; margin-left:8px;}
  .topbar .right{display:flex; align-items:center; gap:14px; font-size:0.8rem; color:#6b7280;}
  .topbar .who{text-align:right; line-height:1.25;}
  .topbar .who b{color:#1f2937;}
  .topbar .mdl{background:#eafaf1; color:#1e8449; font-weight:600;
               padding:4px 12px; border-radius:20px; font-size:0.78rem; white-space:nowrap;}
  header[data-testid="stHeader"]{display:none;}
  .block-container{padding-top:1.2rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_train(cols):
    if not os.path.exists(DATA_PATH):
        return None
    try:
        return pd.read_csv(DATA_PATH)[cols].values
    except Exception:
        return None


def pill(level):
    cls = {"High":"p-red","Elevated":"p-amber","Medium":"p-amber",
           "Low":"p-green","Neutral":"p-green"}.get(level,"p-grey")
    return f"<span class='pill {cls}'>{level}</span>"


def gauge(prob):
    import matplotlib.patches as mp
    fig, ax = plt.subplots(figsize=(3.6,2.1))
    for a0,a1,col in [(0,60,"#1e8449"),(60,120,"#e67e22"),(120,180,"#c0392b")]:
        ax.add_patch(mp.Wedge((0,0),1,180-a1,180-a0,width=0.34,facecolor=col,alpha=0.9))
    ang=np.pi*(1-prob)
    ax.plot([0,0.8*np.cos(ang)],[0,0.8*np.sin(ang)],lw=3,color="#2c3e50")
    ax.add_patch(mp.Circle((0,0),0.05,color="#2c3e50"))
    ax.text(0,-0.18,f"{prob*100:.0f}%",ha="center",fontsize=17,weight="bold")
    ax.set_xlim(-1.2,1.2); ax.set_ylim(-0.35,1.15); ax.axis("off")
    ax.set_aspect("equal"); plt.tight_layout()
    return fig


def shap_fig(model, X1, cols):
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(X1, check_additivity=False)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.array(sv)
    if sv.ndim == 3:            # newer SHAP: (samples, features, classes)
        sv = sv[:, :, 1]
    vals = np.array(sv[0]).ravel()[:len(cols)]
    idx = np.argsort(np.abs(vals))[-10:]
    fig, ax = plt.subplots(figsize=(6,4))
    colors = ["#c0392b" if v>0 else "#2980b9" for v in vals[idx]]
    ax.barh([cols[i] for i in idx], vals[idx], color=colors)
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("← legitimate    SHAP contribution    phishing →")
    ax.set_title("SHAP — why this decision")
    plt.tight_layout()
    return fig


def lime_fig(model, X1, cols, Xtrain):
    ex = LimeTabularExplainer(Xtrain, feature_names=cols,
                              class_names=["Legitimate","Phishing"],
                              mode="classification", discretize_continuous=True)
    e = ex.explain_instance(X1.values[0], model.predict_proba, num_features=8)
    fig = e.as_pyplot_figure(); fig.set_size_inches(6,4)
    plt.title("LIME — local explanation"); plt.tight_layout()
    return fig


def main():
    art = load_model()

    model_label = (f"Model: {art['model_name']}" if art else "Run train.py first")
    mdl_class = "mdl" if art else "pill p-red"
    st.markdown(f"""
    <div class='topbar'>
      <div class='brand'>🛡️ URL Detector <small>Adaptive &amp; Explainable ML</small></div>
      <div class='right'>
        <div class='who'><b>Muhammad Tabish</b><br>Student ID: 77655620</div>
        <span class='{mdl_class}'>{model_label}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Phishing URL threat analysis")
    st.caption("String-based analysis only. URLs are never visited.")

    url = st.text_input("URL to analyse",
                        placeholder="http://secure-login-paypal.xyz/verify@account")
    go = st.button("🔍 Analyse", type="primary")

    if not go or not url.strip():
        st.info("Enter a URL and click Analyse.")
        p = os.path.join(BASE,"models","shap_bar.png")
        if os.path.exists(p):
            st.markdown("#### Global feature importance (SHAP)")
            st.image(p)
        return

    url = url.strip()
    a = full_analysis(url)
    ssl, dns, geo, tok, thr = a["ssl"], a["dns"], a["geo"], a["tokens"], a["threat"]

    # ---- current URL banner ----
    st.markdown(f"<div class='card'><div style='font-size:0.8rem;color:#6b7280'>"
                f"Current URL</div><div class='mono'>{url}</div></div>",
                unsafe_allow_html=True)

    # ---- top stat row ----
    X1 = None; is_phish = None; proba = None
    if art:
        model, cols = art["model"], art["feature_cols"]
        X1 = pd.DataFrame([extract_features(url)])[cols]
        proba = model.predict_proba(X1)[0]
        is_phish = model.predict(X1)[0] == 1

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if is_phish is None:
            v = "—"
        else:
            v = ("<span class='pill p-red'>Phishing</span>" if is_phish
                 else "<span class='pill p-green'>Legitimate</span>")
        st.markdown(f"<div class='stat'><div class='lbl'>ML verdict</div>"
                    f"<div style='margin-top:6px'>{v}</div></div>", unsafe_allow_html=True)
    with c2:
        pv = f"{proba[1]*100:.0f}%" if proba is not None else "—"
        st.markdown(f"<div class='stat'><div class='lbl'>Phishing probability</div>"
                    f"<div class='val'>{pv}</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='stat'><div class='lbl'>Threat score</div>"
                    f"<div class='val'>{thr['score']}<span style='font-size:0.9rem;color:#6b7280'>/100</span></div></div>",
                    unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='stat'><div class='lbl'>Threat level</div>"
                    f"<div style='margin-top:6px'>{pill(thr['level'])}</div></div>",
                    unsafe_allow_html=True)

    if proba is not None:
        g1, g2 = st.columns([1,2])
        with g1:
            st.pyplot(gauge(proba[1])); plt.close()
        with g2:
            st.markdown("**Threat score breakdown**")
            if thr["reasons"]:
                st.markdown("\n".join(f"- **+{p}** {m}" for p,m in thr["reasons"]))
            else:
                st.markdown("- No risk indicators triggered")

    st.markdown("---")

    # ---- analysis cards ----
    st.markdown("#### Detailed analysis")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.markdown(f"<div class='card'><b>🔒 SSL</b><br>{ssl['verdict']}<br>"
                    f"{pill(ssl['risk'])}</div>", unsafe_allow_html=True)
    with d2:
        st.markdown(f"<div class='card'><b>🌐 DNS</b><br>{dns['tld']} · "
                    f"{dns['num_subdomains']} sub<br>{pill(dns['risk'])}</div>",
                    unsafe_allow_html=True)
    with d3:
        st.markdown(f"<div class='card'><b>📍 Region</b><br>{geo['inferred_region']}<br>"
                    f"{pill(geo['risk'])}</div>", unsafe_allow_html=True)
    with d4:
        toks = ", ".join(tok["suspicious_tokens"][:3]) or "none"
        st.markdown(f"<div class='card'><b>🔑 Tokens</b><br>{toks}<br>"
                    f"{pill(tok['risk'])}</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ---- explanations: SHAP + LIME ----
    if art and X1 is not None:
        st.markdown("#### Explainability")
        e1, e2 = st.columns(2)
        with e1:
            try:
                st.pyplot(shap_fig(model, X1, cols)); plt.close()
            except Exception as ex:
                st.warning(f"SHAP unavailable: {ex}")
        with e2:
            Xtr = load_train(cols)
            if Xtr is None:
                st.info("LIME needs training data. Run `python prepare_data.py` "
                        "then `python train.py`, and keep data/dataset.csv in place.")
            else:
                try:
                    st.pyplot(lime_fig(model, X1, cols, Xtr)); plt.close()
                except Exception as ex:
                    st.warning(f"LIME could not render: {ex}")

        with st.expander("📊 Full feature table"):
            feats = extract_features(url)
            st.dataframe(pd.DataFrame(feats.items(), columns=["Feature","Value"]),
                         use_container_width=True, hide_index=True)

    st.caption("Adaptive & Explainable ML Framework · Muhammad Tabish · 77655620 · "
               "Analysis is string-derived; URLs are not visited.")


if __name__ == "__main__":
    main()
