import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tensorflow.keras.models import load_model

from .config import DECISION_THRESHOLD, RESULTS_DIR
from .preprocessing import build_test_generator
from .predict import SAVED_MODEL_PATH, load_detection_model


def _predict(model_path: Path, generator) -> np.ndarray:
    if model_path.is_dir():
        _, signature, _ = load_detection_model(model_path)
        probabilities: list[float] = []
        for batch_index in range(len(generator)):
            images, _ = generator[batch_index]
            outputs = signature(image=tf.convert_to_tensor(images, dtype=tf.float32))
            probabilities.extend(
                np.asarray(outputs["probability_real"]).reshape(-1).tolist()
            )
        return np.asarray(probabilities, dtype=np.float32)

    model = tf.keras.models.load_model(model_path, compile=False)
    return model.predict(generator, verbose=0).ravel()


def evaluate(model_path: str | Path, batch_size: int) -> dict:
    output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = build_test_generator(batch_size)
    probabilities = _predict(Path(model_path), generator)
    y_true = generator.classes
    y_pred = (probabilities >= DECISION_THRESHOLD).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else 0.0
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["fake", "real"],
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "threshold": DECISION_THRESHOLD,
        "samples": int(len(y_true)),
        "test_fake_count": int(np.sum(y_true == 0)),
        "test_real_count": int(np.sum(y_true == 1)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": matrix.tolist(),
        "per_class": {"fake": report["fake"], "real": report["real"]},
        "macro_avg": report["macro avg"],
        "weighted_avg": report["weighted avg"],
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (output_dir / "classification_report.txt").write_text(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["fake", "real"],
            zero_division=0,
        ),
        encoding="utf-8",
    )
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.imshow(matrix, cmap="Blues")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticks([0, 1], ["fake", "real"])
    axis.set_yticks([0, 1], ["fake", "real"])
    for row in range(2):
        for column in range(2):
            axis.text(column, row, matrix[row, column], ha="center", va="center")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(figure)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the classifier on the untouched test set.")
    parser.add_argument("--model", type=Path, default=SAVED_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.model, args.batch_size), indent=2))


if __name__ == "__main__":
    main()
