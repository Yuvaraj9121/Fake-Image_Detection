import argparse
import json
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from .config import BEST_MODEL_PATH, BATCH_SIZE, MODEL_DIR, RANDOM_SEED, RESULTS_DIR
from .model import build_model
from .preprocessing import build_train_generator, build_validation_generator


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train(epochs: int, batch_size: int) -> dict:
    set_seed(RANDOM_SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_generator = build_train_generator(batch_size)
    validation_generator = build_validation_generator(batch_size)
    model = build_model()
    callbacks = [
        ModelCheckpoint(str(BEST_MODEL_PATH), monitor="val_loss", save_best_only=True),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ]
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=callbacks,
    )
    history_data = {key: [float(value) for value in values] for key, values in history.history.items()}
    history_data["best_epoch"] = int(np.argmin(history_data["val_loss"]) + 1)
    (RESULTS_DIR / "training_history.json").write_text(json.dumps(history_data, indent=2) + "\n", encoding="utf-8")
    return history_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the real/fake face classifier.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    history = train(args.epochs, args.batch_size)
    print(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
