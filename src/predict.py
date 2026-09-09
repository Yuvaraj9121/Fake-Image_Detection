import argparse
from pathlib import Path

import numpy as np

from .config import BEST_MODEL_PATH, CLASS_NAMES, DECISION_THRESHOLD
from .preprocessing import preprocess_image


def load_detection_model(model_path: str | Path | None = None):
    path = Path(model_path) if model_path else BEST_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Detection model was not found: {path}")
    try:
        from tensorflow.keras.models import load_model

        return load_model(path, compile=False), path
    except (OSError, ValueError, RuntimeError) as error:
        raise RuntimeError(f"Could not load detection model at {path}: {error}") from error


def predict_image(
    image_path: str | Path,
    model_path: str | Path | None = None,
    model=None,
) -> dict:
    image = preprocess_image(image_path)
    loaded_path = Path(model_path) if model_path else BEST_MODEL_PATH
    if model is None:
        model, loaded_path = load_detection_model(model_path)
    probability = float(model.predict(np.expand_dims(image, axis=0), verbose=0)[0][0])
    class_name = CLASS_NAMES[1] if probability >= DECISION_THRESHOLD else CLASS_NAMES[0]
    confidence = probability if class_name == CLASS_NAMES[1] else 1.0 - probability
    return {
        "path": str(image_path),
        "model_path": str(loaded_path),
        "predicted_class": class_name.upper(),
        "probability_real": probability,
        "confidence": confidence,
        "threshold": DECISION_THRESHOLD,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict whether an image is real or fake.")
    parser.add_argument("image")
    parser.add_argument("--model")
    args = parser.parse_args()
    try:
        print(predict_image(args.image, args.model), flush=True)
    except (OSError, ValueError, FileNotFoundError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
