from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image

from .config import (
    BATCH_SIZE,
    CLASS_NAMES,
    IMAGE_SIZE,
    RANDOM_SEED,
    TEST_DIR,
    TRAIN_DIR,
    VALIDATION_SPLIT,
)


def preprocess_image(image_path: str | Path | BinaryIO) -> np.ndarray:
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        array = np.asarray(rgb_image, dtype=np.float32)
    return array


def _random_contrast(image: np.ndarray) -> np.ndarray:
    import tensorflow as tf

    tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    return tf.image.random_contrast(tensor, lower=0.9, upper=1.1, seed=RANDOM_SEED).numpy()


def build_train_generator(batch_size: int = BATCH_SIZE):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    generator = ImageDataGenerator(
        rotation_range=8.0,
        zoom_range=0.08,
        horizontal_flip=True,
        preprocessing_function=_random_contrast,
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
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    generator = ImageDataGenerator(
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
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    generator = ImageDataGenerator()
    return generator.flow_from_directory(
        TEST_DIR,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        classes=list(CLASS_NAMES),
        shuffle=False,
    )
