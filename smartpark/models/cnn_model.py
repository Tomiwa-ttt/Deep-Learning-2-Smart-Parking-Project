"""
cnn_model.py

CNN architecture for classifying a cropped parking-spot image as
Empty (0) or Occupied (1). Matches the standard approach used in the
PKLot / CNRPark-EXT research: a lightweight CNN applied per-spot-patch,
not full-scene detection.
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_cnn(input_size=64, channels=3):
    """Small CNN, cheap enough to train on CPU and to run in real time
    on dozens of spot crops per frame."""
    model = keras.Sequential([
        keras.Input(shape=(input_size, input_size, channels)),

        layers.Rescaling(1.0 / 255),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(1, activation="sigmoid"),  # Empty=0, Occupied=1
    ], name="parking_spot_cnn")

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_mobilenet_transfer(input_size=96, channels=3):
    """Transfer-learning alternative (MobileNetV2 backbone). Better accuracy
    with less training data once you move to a real dataset like PKLot/
    CNRPark-EXT; heavier per-inference than the small CNN above."""
    base = keras.applications.MobileNetV2(
        input_shape=(input_size, input_size, channels),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # freeze backbone; unfreeze top layers later to fine-tune

    model = keras.Sequential([
        keras.Input(shape=(input_size, input_size, channels)),
        layers.Rescaling(1.0 / 127.5, offset=-1),  # MobileNetV2 preprocessing
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ], name="parking_spot_mobilenet")

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    m = build_cnn()
    m.summary()
