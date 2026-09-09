from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from .config import (
    BATCH_SIZE,
    CLASS_NAMES,
    IMAGE_SIZE,
    RANDOM_SEED,
    TRAIN_DIR,
    VALIDATION_SPLIT,
)


def preprocess_image(image_path: str | Path | BinaryIO) -> np.ndarray:
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB").resize(IMAGE_SIZE)
        array = np.asarray(rgb_image, dtype=np.float32) / 255.0
    return array


def build_train_generator(batch_size: int = BATCH_SIZE):
    generator = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=10.0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=VALIDATION_SPLIT,
    )
    return generator.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        subset="training",
        classes=list(CLASS_NAMES),
        seed=RANDOM_SEED,
        shuffle=True,
    )


def build_validation_generator(batch_size: int = BATCH_SIZE):
    generator = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=VALIDATION_SPLIT,
    )
    return generator.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        subset="validation",
        classes=list(CLASS_NAMES),
        seed=RANDOM_SEED,
        shuffle=False,
    )


def build_test_generator(batch_size: int = BATCH_SIZE):
    generator = ImageDataGenerator(rescale=1.0 / 255.0)
    return generator.flow_from_directory(
        TRAIN_DIR.parent / "test",
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        classes=list(CLASS_NAMES),
        shuffle=False,
    )
