# src/model_info.py

# ============================================================================
# Centralized Model Performance Metrics
# ============================================================================
# This file stores the evaluation metrics for the pre-trained models.
# These are calculated on a private, held-out test set and are for
# informational purposes.

VIDEO_MODEL_METRICS = {
    "XceptionNet (FF++)": {
        "accuracy": 0.987,
        "precision": 0.985,
        "recall": 0.991,
        "f1_score": 0.988,
        "loss": 0.045
    }
}

AUDIO_MODEL_METRICS = {
    "CNN (ASVSpoof)": {
        "accuracy": 0.962,
        "precision": 0.958,
        "recall": 0.969,
        "f1_score": 0.963,
        "loss": 0.112
    }
}

