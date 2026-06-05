from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer


LOG_TYPES = ("sys", "kernel", "auth", "ovs")
PLATFORMS = ("linux", "macos", "windows", "unknown")
FEATURE_VECTOR_SIZE = 12
KEYWORD_CONFIDENCE_THRESHOLD = 0.8

TIMESTAMP_PATTERNS = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}"),
    re.compile(r"\b[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}"),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}:\d{2}"),
]

LEVEL_PATTERN = re.compile(r"\b(?:ERROR|ERR|WARN|WARNING|INFO|DEBUG|TRACE|CRITICAL|FATAL)\b", re.IGNORECASE)

TYPE_KEYWORDS = {
    "sys": ["syslog", "systemd", "service", "daemon", "journal"],
    "kernel": ["kernel", "dmesg", "irq", "segfault", "panic", "module"],
    "auth": ["auth", "pam", "sudo", "login", "sshd", "failed password"],
    "ovs": ["openvswitch", "ovs-vswitchd", "ovsdb", "bridge", "datapath"],
}

PLATFORM_KEYWORDS = {
    "linux": ["systemd", "journalctl", "kernel", "sshd", "/var/log"],
    "macos": ["darwin", "launchd", "asl", "macos", "com.apple"],
    "windows": ["eventlog", "powershell", "win32", "microsoft", "service control manager"],
}


@dataclass
class PredictionResult:
    is_log_file: bool
    confidence: float
    log_type: Optional[str]
    log_type_confidence: Optional[float]
    platform: Optional[str]
    platform_confidence: Optional[float]


def _safe_text(value: str) -> str:
    return value if isinstance(value, str) else ""


def extract_features(text: str) -> np.ndarray:
    content = _safe_text(text)
    lines = [line for line in content.splitlines() if line.strip()]
    if not lines:
        return np.zeros(FEATURE_VECTOR_SIZE, dtype=float)

    line_count = len(lines)
    timestamp_hits = sum(any(pattern.search(line) for pattern in TIMESTAMP_PATTERNS) for line in lines)
    level_hits = sum(bool(LEVEL_PATTERN.search(line)) for line in lines)
    structured_hits = sum(bool(re.search(r"[:\[\]\-]", line)) for line in lines)

    lower_text = content.lower()
    type_scores = [sum(lower_text.count(keyword) for keyword in keywords) for keywords in TYPE_KEYWORDS.values()]
    platform_scores = [sum(lower_text.count(keyword) for keyword in keywords) for keywords in PLATFORM_KEYWORDS.values()]

    avg_line_len = sum(len(line) for line in lines) / line_count
    digit_ratio = sum(char.isdigit() for char in content) / max(1, len(content))
    uppercase_ratio = sum(char.isupper() for char in content) / max(1, len(content))

    return np.array(
        [
            line_count,
            avg_line_len,
            timestamp_hits / line_count,
            level_hits / line_count,
            structured_hits / line_count,
            digit_ratio,
            uppercase_ratio,
            *type_scores,
            *platform_scores,
        ],
        dtype=float,
    )


def extract_feature_matrix(texts: Sequence[str]) -> np.ndarray:
    return np.vstack([extract_features(_safe_text(text)) for text in texts])


def _keyword_prediction(text: str, keyword_map: Dict[str, List[str]]) -> Optional[Tuple[str, float]]:
    lower_text = text.lower()
    scores: Dict[str, int] = {
        label: sum(lower_text.count(keyword) for keyword in keywords)
        for label, keywords in keyword_map.items()
    }
    best_label, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score <= 0:
        return None
    total = sum(scores.values()) or 1
    return best_label, float(best_score / total)


def _blend_with_keywords(
    predicted_label: str,
    predicted_confidence: float,
    text: str,
    keyword_map: Dict[str, List[str]],
) -> Tuple[str, float]:
    keyword_result = _keyword_prediction(text, keyword_map)
    if keyword_result is None:
        return predicted_label, predicted_confidence

    heuristic_label, keyword_confidence = keyword_result
    if keyword_confidence >= KEYWORD_CONFIDENCE_THRESHOLD and keyword_confidence >= predicted_confidence:
        predicted_label = heuristic_label
    predicted_confidence = max(predicted_confidence, keyword_confidence)
    return predicted_label, predicted_confidence


def _build_pipeline() -> Pipeline:
    combined_features = FeatureUnion(
        [
            (
                "word_tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=4000,
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=1,
                    max_features=3000,
                ),
            ),
            (
                "stats",
                FunctionTransformer(extract_feature_matrix, validate=False),
            ),
        ]
    )

    return Pipeline(
        [
            ("features", combined_features),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def train_models(samples: Sequence[Dict[str, str]]) -> Dict[str, Pipeline]:
    if not samples:
        raise ValueError("Training data is empty.")

    texts = [_safe_text(sample.get("text", "")) for sample in samples]
    is_log_labels = np.array([int(sample.get("is_log", 0)) for sample in samples])

    binary_model = _build_pipeline()
    binary_model.fit(texts, is_log_labels)

    log_samples = [sample for sample in samples if int(sample.get("is_log", 0)) == 1]
    if not log_samples:
        raise ValueError("No log samples present for multi-class training.")

    log_texts = [_safe_text(sample.get("text", "")) for sample in log_samples]
    log_type_labels = np.array([sample.get("log_type", "sys") for sample in log_samples])
    platform_labels = np.array([sample.get("platform", "unknown") for sample in log_samples])

    log_type_model = _build_pipeline()
    log_type_model.fit(log_texts, log_type_labels)

    platform_model = _build_pipeline()
    platform_model.fit(log_texts, platform_labels)

    return {
        "binary": binary_model,
        "log_type": log_type_model,
        "platform": platform_model,
    }


def save_models(models: Dict[str, Pipeline], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, output_path)


def load_jsonl_dataset(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []

    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


class LogFileClassifier:
    def __init__(self, model_path: str = "models/log_classifier.joblib") -> None:
        self.model_path = Path(model_path)
        self.models: Optional[Dict[str, Pipeline]] = None

    def load(self) -> bool:
        if not self.model_path.exists():
            return False
        self.models = joblib.load(self.model_path)
        return True

    def predict(self, text: str) -> PredictionResult:
        if self.models is None:
            raise RuntimeError("Model is not loaded. Train or load a model before prediction.")

        cleaned_text = _safe_text(text)
        if not cleaned_text.strip():
            return PredictionResult(False, 0.0, None, None, None, None)

        binary_model = self.models["binary"]
        binary_probs = binary_model.predict_proba([cleaned_text])[0]
        binary_classes = binary_model.classes_
        class_to_prob = {int(cls): float(prob) for cls, prob in zip(binary_classes, binary_probs)}

        log_probability = class_to_prob.get(1, 0.0)
        is_log_file = log_probability >= 0.5

        if not is_log_file:
            return PredictionResult(False, 1.0 - log_probability, None, None, None, None)

        log_type_model = self.models["log_type"]
        log_type_probs = log_type_model.predict_proba([cleaned_text])[0]
        log_type_index = int(np.argmax(log_type_probs))
        log_type = str(log_type_model.classes_[log_type_index])
        log_type_confidence = float(log_type_probs[log_type_index])
        log_type, log_type_confidence = _blend_with_keywords(
            log_type, log_type_confidence, cleaned_text, TYPE_KEYWORDS
        )

        platform_model = self.models["platform"]
        platform_probs = platform_model.predict_proba([cleaned_text])[0]
        platform_index = int(np.argmax(platform_probs))
        platform = str(platform_model.classes_[platform_index])
        platform_confidence = float(platform_probs[platform_index])
        platform, platform_confidence = _blend_with_keywords(
            platform, platform_confidence, cleaned_text, PLATFORM_KEYWORDS
        )

        return PredictionResult(
            is_log_file=True,
            confidence=log_probability,
            log_type=log_type,
            log_type_confidence=log_type_confidence,
            platform=platform,
            platform_confidence=platform_confidence,
        )
