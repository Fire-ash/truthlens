#!/usr/bin/env python3
"""TruthLens - AI-powered fake news classification interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import streamlit as st

from detect_fake_news import classify_probability
from model_compat import load_pipeline


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_pipeline_path() -> Path:
    return project_root() / "outputs" / "pipeline.joblib"


def default_metrics_path() -> Path:
    return project_root() / "outputs" / "metrics.json"


@st.cache_resource
def load_model(path: str):
    return load_pipeline(path)


@st.cache_data
def load_metrics(path: str) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def word_count(text: str) -> int:
    return len(text.split())


def get_confidence(prob_fake: float) -> float:
    return max(prob_fake, 1.0 - prob_fake)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--pipeline", default=str(default_pipeline_path()))
    args, _ = parser.parse_known_args()

    pipeline_path = Path(args.pipeline).resolve()
    metrics_path = default_metrics_path()

    st.set_page_config(
        page_title="TruthLens",
        page_icon="TL",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1150px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        .hero {
            padding: 1.5rem 0 1rem 0;
        }

        .brand {
            font-size: 2.6rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            margin-bottom: 0.15rem;
        }

        .subtitle {
            color: #6b7280;
            font-size: 1.05rem;
        }

        .result-card {
            padding: 2rem;
            border-radius: 18px;
            border: 1px solid rgba(128,128,128,.25);
            margin-top: 1.5rem;
        }

        .result-label {
            font-size: 3rem;
            font-weight: 800;
            margin: 0;
        }

        .result-score {
            font-size: 1.3rem;
            margin-top: .25rem;
        }

        .metric-card {
            padding: 1.25rem;
            border-radius: 14px;
            border: 1px solid rgba(128,128,128,.25);
            min-height: 110px;
        }

        .metric-name {
            color: #6b7280;
            font-size: .9rem;
        }

        .metric-value {
            font-size: 1.7rem;
            font-weight: 700;
            margin-top: .25rem;
        }

        .section-title {
            font-size: 1.35rem;
            font-weight: 700;
            margin-top: 2rem;
            margin-bottom: .75rem;
        }

        .disclaimer {
            padding: 1rem 1.2rem;
            border-radius: 12px;
            background: rgba(128,128,128,.10);
            margin-top: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if not pipeline_path.exists():
        st.error(
            "The trained model could not be found. "
            "Run the training script before starting the application."
        )
        st.stop()

    pipeline = load_model(str(pipeline_path))
    metrics = load_metrics(metrics_path)

    st.markdown(
        """
        <div class="hero">
            <div class="brand">TruthLens</div>
            <div class="subtitle">
                AI-powered fake news classification and risk analysis
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        '<div class="section-title">News Analysis</div>',
        unsafe_allow_html=True,
    )

    text = st.text_area(
        "News content",
        height=260,
        placeholder=(
            "Paste a news article, report, or headline here..."
        ),
        label_visibility="collapsed",
    )

    count_col, button_col = st.columns([4, 1])

    with count_col:
        st.caption(f"{word_count(text):,} words")

    with button_col:
        analyze = st.button(
            "Analyze News",
            type="primary",
            use_container_width=True,
        )

    with st.expander("Advanced Settings"):
        threshold = st.slider(
            "FAKE decision threshold",
            0.05,
            0.95,
            0.50,
            0.01,
        )

        uncertainty_margin = st.slider(
            "UNCERTAIN band width",
            0.00,
            0.30,
            0.10,
            0.01,
            help=(
                "Predictions close to the decision threshold are "
                "classified as uncertain."
            ),
        )

    if analyze:
        if not text.strip():
            st.warning("Enter some news content before analyzing.")
            st.stop()

        prob_fake = float(pipeline.predict_proba([text])[0, 1])

        label = classify_probability(
            prob_fake,
            threshold,
            uncertainty_margin,
        )

        confidence = get_confidence(prob_fake)

        if label == "FAKE":
            result_description = "The text is classified as FAKE."
        elif label == "REAL":
            result_description = "The text is classified as REAL."
        else:
            result_description = (
                "The model is close to its decision boundary."
            )

        st.markdown(
            '<div class="section-title">Analysis Result</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">{label}</div>
                <div class="result-score">
                    Classification confidence: <strong>{confidence:.1%}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(
            confidence,
            text=f"Model confidence: {confidence:.1%}",
        )

        st.write(result_description)

        if label == "UNCERTAIN":
            st.warning(
                "The prediction is close to the decision boundary. "
                "Consider providing more context."
            )

        if word_count(text) < 30:
            st.warning(
                "Short inputs provide fewer linguistic features and "
                "may produce less reliable predictions."
            )

    st.markdown(
        '<div class="section-title">Model Performance</div>',
        unsafe_allow_html=True,
    )

    if metrics:
        test = metrics.get("holdout_test", {})

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-name">Accuracy</div>
                    <div class="metric-value">
                        {test.get("accuracy", 0):.1%}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-name">Macro F1</div>
                    <div class="metric-value">
                        {test.get("macro_f1", 0):.1%}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-name">ROC-AUC</div>
                    <div class="metric-value">
                        {test.get("roc_auc", 0):.3f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-name">Test Samples</div>
                    <div class="metric-value">
                        {test.get("classification_report", {})
                        .get("weighted avg", {})
                        .get("support", 0):.0f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="section-title">How the model works</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "The system cleans the input text, converts it into TF-IDF "
        "features, and uses Logistic Regression to estimate the "
        "probability that the text belongs to the FAKE class."
    )

    st.markdown(
        """
        <div class="disclaimer">
            <strong>Important limitation</strong><br>
            TruthLens is a machine-learning text classifier, not an
            independent fact-checking engine. Its predictions reflect
            patterns learned from the training dataset and should be
            verified against reliable primary sources.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
