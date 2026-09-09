from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

from .config import INPUT_SHAPE


def build_model():
    model = Sequential(
        [
            Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=INPUT_SHAPE),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(256, activation="relu"),
            Dense(1, activation="sigmoid"),
        ],
        name="real_fake_face_cnn",
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
