import argparse
import json
import random
import shutil

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam

from .config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    FINAL_MODEL_PATH,
    MODEL_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
)
from .model import build_model
from .preprocessing import build_train_generator, build_validation_generator


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _class_weights(classes: np.ndarray) -> dict[int, float]:
    counts = np.bincount(classes.astype(int), minlength=2)
    total = counts.sum()
    return {index: float(total / (2 * count)) for index, count in enumerate(counts) if count}


def _callbacks(candidate_path: str) -> list:
    return [
        ModelCheckpoint(candidate_path, monitor="val_loss", save_best_only=True, mode="min"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7, mode="min"),
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, mode="min"),
    ]


def _unfreeze_top_layers(model: tf.keras.Model, layer_count: int = 20) -> None:
    backbone = model.get_layer("efficientnetb0")
    backbone.trainable = True
    for layer in backbone.layers[:-layer_count]:
        layer.trainable = False
    for layer in backbone.layers[-layer_count:]:
        layer.trainable = not isinstance(layer, BatchNormalization)


def _merge_history(target: dict[str, list[float]], source: dict[str, list[float]]) -> None:
    for key, values in source.items():
        target.setdefault(key, []).extend(float(value) for value in values)


def _save_history(history_data: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "training_history.json").write_text(
        json.dumps(history_data, indent=2) + "\n", encoding="utf-8"
    )
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history_data["loss"], label="train")
    axes[0].plot(history_data["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history_data["accuracy"], label="train")
    axes[1].plot(history_data["val_accuracy"], label="validation")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "training_history.png", dpi=150)
    plt.close(figure)


def train(epochs: int, fine_tune_epochs: int, batch_size: int) -> dict:
    if epochs < 1 or fine_tune_epochs < 0:
        raise ValueError("epochs must be at least 1 and fine_tune_epochs cannot be negative")

    set_seed(RANDOM_SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    candidate_path = MODEL_DIR / ".best_model_candidate.keras"
    if candidate_path.exists():
        candidate_path.unlink()

    train_generator = build_train_generator(batch_size)
    validation_generator = build_validation_generator(batch_size)
    class_weights = _class_weights(train_generator.classes)
    model = build_model(learning_rate=1e-3)
    combined_history: dict[str, list[float]] = {}

    head_history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=_callbacks(str(candidate_path)),
    )
    _merge_history(combined_history, head_history.history)

    if fine_tune_epochs:
        _unfreeze_top_layers(model)
        model.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        fine_tune_history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=fine_tune_epochs,
            class_weight=class_weights,
            callbacks=_callbacks(str(candidate_path)),
        )
        _merge_history(combined_history, fine_tune_history.history)

    if not candidate_path.exists():
        raise RuntimeError("Training did not produce a validation checkpoint")

    best_epoch = int(np.argmin(combined_history["val_loss"]) + 1)
    combined_history["best_epoch"] = best_epoch
    combined_history["class_weights"] = class_weights
    _save_history(combined_history)
    shutil.copy2(candidate_path, BEST_MODEL_PATH)
    model.save(FINAL_MODEL_PATH)
    candidate_path.unlink()
    return combined_history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the real/fake face classifier.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    print(json.dumps(train(args.epochs, args.fine_tune_epochs, args.batch_size), indent=2))


if __name__ == "__main__":
    main()
