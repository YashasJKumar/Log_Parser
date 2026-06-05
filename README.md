# Log File Classification (ML + Streamlit)

This repository includes a machine-learning pipeline for classifying uploaded files as **log** or **non-log**, and for identifying:
- log type (`sys`, `kernel`, `auth`, `ovs`)
- likely platform (`Linux`, `macOS`, `Windows`)

## Project structure

- `data/` - training dataset (`training_data.jsonl`)
- `models/` - persisted trained model artifacts
- `classifier.py` - feature extraction, model training, inference, and persistence utilities
- `train_model.py` - training script
- `app.py` - Streamlit UI for file upload + random 20 line preview + classification
- `requirements.txt` - Python dependencies

## Features

- Binary classification: log vs non-log
- Multi-class classification: `sys`, `kernel`, `auth`, `ovs`
- Platform classification: `linux`, `macos`, `windows`
- Confidence scores for every prediction
- Graceful handling for empty/unreadable files and missing model artifacts

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train the model

```bash
python train_model.py
```

Optional arguments:

```bash
python train_model.py --data data/training_data.jsonl --output models/log_classifier.joblib
```

## Run the Streamlit app

```bash
streamlit run app.py
```

## Training data format

`data/training_data.jsonl` expects one JSON object per line:

```json
{"text":"<file or line sample>","is_log":1,"log_type":"sys","platform":"linux"}
```

Notes:
- For non-log rows, use `is_log: 0` and keep `log_type: "none"`, `platform: "unknown"`.
- Multi-class (`log_type`, `platform`) models are trained only with rows where `is_log = 1`.
