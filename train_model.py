from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from classifier import save_models, train_models


def load_training_samples(data_path: Path) -> List[Dict[str, str]]:
    if not data_path.exists():
        raise FileNotFoundError(f"Training data file not found: {data_path}")

    samples: List[Dict[str, str]] = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))

    if not samples:
        raise ValueError(f"Training data file is empty: {data_path}")

    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Train log classifier model")
    parser.add_argument("--data", default="data/training_data.jsonl", help="Path to JSONL training data")
    parser.add_argument("--output", default="models/log_classifier.joblib", help="Path to save trained model")
    args = parser.parse_args()

    data_path = Path(args.data)
    output_path = Path(args.output)

    samples = load_training_samples(data_path)

    labels = [int(sample.get("is_log", 0)) for sample in samples]

    train_subset, test_subset = train_test_split(
        samples,
        test_size=0.25,
        random_state=42,
        stratify=labels if len(set(labels)) > 1 else None,
    )

    models = train_models(train_subset)
    save_models(models, output_path)

    binary_model = models["binary"]
    x_test = [sample.get("text", "") for sample in test_subset]
    y_test = [int(sample.get("is_log", 0)) for sample in test_subset]
    predictions = binary_model.predict(x_test)

    print(f"Training samples: {len(train_subset)}")
    print(f"Saved model to: {output_path}")
    print(f"Binary accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(classification_report(y_test, predictions, zero_division=0))


if __name__ == "__main__":
    main()
