import argparse
from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model

from .config import BEST_MODEL_PATH, CLASS_NAMES, ORIGINAL_CHECKPOINT
from .preprocessing import preprocess_image


def load_detection_model(model_path: str | Path | None = None):
    path = Path(model_path) if model_path else BEST_MODEL_PATH
    if not path.exists() and ORIGINAL_CHECKPOINT.exists():
        path = ORIGINAL_CHECKPOINT
    if not path.exists():
        raise FileNotFoundError(f"No model found at {path}")
    return load_model(path, compile=False), path


def predict_image(image_path: str | Path, model_path: str | Path | None = None) -> dict:
    image = preprocess_image(image_path)
    model, loaded_path = load_detection_model(model_path)
    probability = float(model.predict(np.expand_dims(image, axis=0), verbose=0)[0][0])
    class_name = CLASS_NAMES[1] if probability >= 0.5 else CLASS_NAMES[0]
    confidence = probability if class_name == CLASS_NAMES[1] else 1.0 - probability
    return {
        "path": str(image_path),
        "model": str(loaded_path),
        "class": class_name,
        "probability_real": probability,
        "confidence": confidence,
        "threshold": 0.5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict whether an image is real or fake.")
    parser.add_argument("image")
    parser.add_argument("--model")
    args = parser.parse_args()
    try:
        print(predict_image(args.image, args.model))
    except (OSError, ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
