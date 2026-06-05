from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer


LOG_TYPES = ("sys", "kernel", "auth", "ovs")
PLATFORMS = ("linux", "macos", "windows", "unknown")

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
        return np.zeros(12, dtype=float)

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


def _keyword_prediction(text: str, keyword_map: Dict[str, List[str]]) -> Optional[tuple[str, float]]:
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


def build_default_dataset() -> List[Dict[str, str]]:
    return [
        {"text": "Jan 10 10:01:14 host1 systemd[1]: Started Daily apt download activities.", "is_log": 1, "log_type": "sys", "platform": "linux"},
        {"text": "2025-01-11 09:16:52 INFO service[2231]: rotating syslog file", "is_log": 1, "log_type": "sys", "platform": "linux"},
        {"text": "May 02 14:22:03 MacBook-Pro com.apple.xpc.launchd[1] <Notice>: Service exited", "is_log": 1, "log_type": "sys", "platform": "macos"},
        {"text": "2025-05-02 12:01:03 kernel: [13092.120] usb 2-1: reset high-speed USB device", "is_log": 1, "log_type": "kernel", "platform": "linux"},
        {"text": "Jun 04 11:09:45 node1 kernel: BUG: soft lockup - CPU#2 stuck for 26s", "is_log": 1, "log_type": "kernel", "platform": "linux"},
        {"text": "2025-06-04 11:42:14 WARN kernel32 module fault detected in win32 stack", "is_log": 1, "log_type": "kernel", "platform": "windows"},
        {"text": "Jan 25 08:17:41 server sshd[19221]: Failed password for invalid user admin from 10.0.0.3", "is_log": 1, "log_type": "auth", "platform": "linux"},
        {"text": "2025-01-25 08:18:12 INFO sudo: pam_unix(sudo:session): session opened for user root", "is_log": 1, "log_type": "auth", "platform": "linux"},
        {"text": "06/01/2025 07:01:11 ERROR EventLog: Logon failure for user SERVICE_ACCOUNT", "is_log": 1, "log_type": "auth", "platform": "windows"},
        {"text": "2025-03-10T10:11:22.120Z|00035|ovs-vswitchd|WARN|bridge br-int: dropped packet", "is_log": 1, "log_type": "ovs", "platform": "linux"},
        {"text": "2025-03-10T10:12:22.120Z|00036|ovsdb-server|INFO|compaction complete", "is_log": 1, "log_type": "ovs", "platform": "linux"},
        {"text": "2025-03-10 12:41:00 INFO openvswitch datapath reconnect on darwin test host", "is_log": 1, "log_type": "ovs", "platform": "macos"},
        {"text": "Shopping list: milk, bread, eggs, apples and cereal", "is_log": 0, "log_type": "none", "platform": "unknown"},
        {"text": "def calculate_total(items):\n    return sum(items)", "is_log": 0, "log_type": "none", "platform": "unknown"},
        {"text": "Meeting notes: finalize quarterly roadmap and assign action items.", "is_log": 0, "log_type": "none", "platform": "unknown"},
        {"text": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.", "is_log": 0, "log_type": "none", "platform": "unknown"},
        {"text": "This text document explains how to install an application manually.", "is_log": 0, "log_type": "none", "platform": "unknown"},
    ]


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
        keyword_log_type = _keyword_prediction(cleaned_text, TYPE_KEYWORDS)
        if keyword_log_type is not None:
            log_type, keyword_confidence = keyword_log_type
            log_type_confidence = max(log_type_confidence, keyword_confidence)

        platform_model = self.models["platform"]
        platform_probs = platform_model.predict_proba([cleaned_text])[0]
        platform_index = int(np.argmax(platform_probs))
        platform = str(platform_model.classes_[platform_index])
        platform_confidence = float(platform_probs[platform_index])
        keyword_platform = _keyword_prediction(cleaned_text, PLATFORM_KEYWORDS)
        if keyword_platform is not None:
            platform, keyword_confidence = keyword_platform
            platform_confidence = max(platform_confidence, keyword_confidence)

        return PredictionResult(
            is_log_file=True,
            confidence=log_probability,
            log_type=log_type,
            log_type_confidence=log_type_confidence,
            platform=platform,
            platform_confidence=platform_confidence,
        )
