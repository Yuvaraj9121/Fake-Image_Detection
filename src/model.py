from tensorflow.keras import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam

from .config import INPUT_SHAPE


def build_model(learning_rate: float = 1e-3) -> Model:
    inputs = Input(shape=INPUT_SHAPE, name="image")
    backbone = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=INPUT_SHAPE,
    )
    backbone.trainable = False
    features = backbone(inputs, training=False)
    features = GlobalAveragePooling2D(name="global_average_pooling")(features)
    features = Dropout(0.3, name="dropout")(features)
    outputs = Dense(1, activation="sigmoid", name="probability_real")(features)
    model = Model(inputs, outputs, name="efficientnetb0_real_fake_classifier")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
