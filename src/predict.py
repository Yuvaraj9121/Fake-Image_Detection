from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import tensorflow as tf

from .config import CLASS_NAMES, DECISION_THRESHOLD, ROOT_DIR
from .preprocessing import preprocess_image


# The deployment model is a TensorFlow SavedModel.
# This avoids the Keras 2.x -> Keras 3.x deserialization problem.
SAVED_MODEL_PATH = ROOT_DIR / "models" / "real_fake_savedmodel"

_MODEL: Any | None = None
_SIGNATURE: Any | None = None


def load_detection_model(
    model_path: str | Path | None = None,
) -> tuple[Any, Any, Path]:
    """
    Load the TensorFlow SavedModel and its serving signature.

    The model is loaded once and reused for subsequent predictions.
    """
    global _MODEL, _SIGNATURE

    path = Path(model_path) if model_path else SAVED_MODEL_PATH

    if not path.is_absolute():
        path = ROOT_DIR / path

    if not path.exists():
        raise FileNotFoundError(
            f"Detection model was not found: {path}"
        )

    if not path.is_dir():
        raise RuntimeError(
            f"Detection model must be a SavedModel directory: {path}"
        )

    try:
        loaded_model = tf.saved_model.load(str(path))

        if "serving_default" not in loaded_model.signatures:
            available_signatures = list(loaded_model.signatures.keys())
            raise RuntimeError(
                "The SavedModel does not contain the required "
                f"'serving_default' signature. Available signatures: "
                f"{available_signatures}"
            )

        signature = loaded_model.signatures["serving_default"]

        _MODEL = loaded_model
        _SIGNATURE = signature

        return loaded_model, signature, path

    except FileNotFoundError:
        raise
    except Exception as error:
        raise RuntimeError(
            f"Could not load detection model at {path}: {error}"
        ) from error


def _get_model_output(
    signature: Any,
    image_batch: np.ndarray,
) -> float:
    """
    Run inference and extract the probability of the REAL class.
    """
    tensor = tf.convert_to_tensor(image_batch, dtype=tf.float32)

    try:
        outputs = signature(image=tensor)
    except TypeError:
        # Fallback for a SavedModel signature whose input name differs.
        input_names = list(signature.structured_input_signature[1].keys())

        if len(input_names) != 1:
            raise RuntimeError(
                "Unable to determine the SavedModel input name."
            )

        outputs = signature(**{input_names[0]: tensor})

    if isinstance(outputs, dict):
        if "probability_real" in outputs:
            probability_tensor = outputs["probability_real"]
        elif len(outputs) == 1:
            probability_tensor = next(iter(outputs.values()))
        else:
            raise RuntimeError(
                "Unable to determine the REAL probability from model outputs: "
                f"{list(outputs.keys())}"
            )
    else:
        probability_tensor = outputs

    probability = float(np.asarray(probability_tensor).reshape(-1)[0])

    if not 0.0 <= probability <= 1.0:
        raise RuntimeError(
            f"Model returned an invalid probability: {probability}"
        )

    return probability


def predict_image(
    image_path: str | Path | BinaryIO,
    model: Any | None = None,
    signature: Any | None = None,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Predict whether an image is REAL or FAKE.

    Returns:
        A dictionary containing:
        - path
        - model_path
        - predicted_class
        - probability_real
        - confidence
        - threshold
    """
    try:
        image = preprocess_image(image_path)
    except Exception as error:
        raise ValueError(
            f"Could not process input image: {error}"
        ) from error

    image_batch = np.expand_dims(image, axis=0)

    loaded_path: Path

    if model is None or signature is None:
        model, signature, loaded_path = load_detection_model(model_path)
    else:
        loaded_path = (
            Path(model_path)
            if model_path
            else SAVED_MODEL_PATH
        )

    probability_real = _get_model_output(
        signature,
        image_batch,
    )

    if probability_real >= DECISION_THRESHOLD:
        class_name = CLASS_NAMES[1]
        confidence = probability_real
    else:
        class_name = CLASS_NAMES[0]
        confidence = 1.0 - probability_real

    return {
        "path": str(image_path),
        "model_path": str(loaded_path),
        "predicted_class": class_name.upper(),
        "probability_real": probability_real,
        "confidence": confidence,
        "threshold": DECISION_THRESHOLD,
    }


def main() -> None:
    """Run single-image prediction from the command line."""
    parser = argparse.ArgumentParser(
        description="Predict whether an image is REAL or FAKE."
    )
    parser.add_argument(
        "image",
        type=Path,
        help="Path to the image to classify.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Optional path to a TensorFlow SavedModel directory.",
    )

    args = parser.parse_args()

    result = predict_image(
        image_path=args.image,
        model_path=args.model,
    )

    print(result)


if __name__ == "__main__":
    main()