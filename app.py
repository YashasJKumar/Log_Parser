from __future__ import annotations

import random
from pathlib import Path
from typing import List

import streamlit as st

from classifier import LogFileClassifier


MODEL_PATH = Path("models/log_classifier.joblib")
MAX_PREVIEW_LINES = 20


def decode_uploaded_file(file_bytes: bytes) -> str:
    if not file_bytes:
        return ""

    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue

    return file_bytes.decode("utf-8", errors="ignore")


def sample_random_lines(content: str, count: int = MAX_PREVIEW_LINES) -> List[str]:
    lines = [line for line in content.splitlines() if line.strip()]
    if not lines:
        return []
    if len(lines) <= count:
        return lines
    return random.sample(lines, count)


def render_prediction(content: str) -> None:
    classifier = LogFileClassifier(str(MODEL_PATH))
    if not classifier.load():
        st.error("Model file not found. Run `python train_model.py` first.")
        return

    try:
        prediction = classifier.predict(content)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    st.subheader("Classification Result")
    st.write(f"**Is log file:** {'Yes' if prediction.is_log_file else 'No'}")
    st.write(f"**Confidence:** {prediction.confidence:.2%}")

    if prediction.is_log_file:
        st.write(f"**Log type:** {prediction.log_type or 'Unknown'}")
        if prediction.log_type_confidence is not None:
            st.write(f"**Log type confidence:** {prediction.log_type_confidence:.2%}")

        st.write(f"**Platform:** {prediction.platform or 'Unknown'}")
        if prediction.platform_confidence is not None:
            st.write(f"**Platform confidence:** {prediction.platform_confidence:.2%}")


def main() -> None:
    st.set_page_config(page_title="Log File Classifier", page_icon="🧠", layout="wide")
    st.title("ML Log File Classifier")
    st.write("Upload any file to preview random lines and classify whether it is a log file.")

    uploaded_file = st.file_uploader("Upload file", type=None)
    if uploaded_file is None:
        st.info("Upload a file to start classification.")
        return

    content = decode_uploaded_file(uploaded_file.getvalue())

    if not content.strip():
        st.warning("The uploaded file is empty or not readable as text.")
        return

    sampled = sample_random_lines(content, MAX_PREVIEW_LINES)
    st.subheader(f"Random {len(sampled)} Lines (up to {MAX_PREVIEW_LINES})")
    if sampled:
        st.code("\n".join(sampled), language="text")
    else:
        st.write("No non-empty lines found.")

    render_prediction(content)


if __name__ == "__main__":
    main()
