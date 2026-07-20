"""
detector_model.py

A small single-shot object detector (YOLOv1-style: one predicted box per
grid cell, trained from scratch) that finds and classifies parking spots
directly in a full, uncropped lot photo -- two classes, "empty_spot" and
"occupied_spot".

Unlike models/cnn_model.py (which classifies a pre-cropped, pre-calibrated
spot patch), this model takes the whole image and outputs a grid of
(objectness, box, class) predictions, so no manual per-camera spot
calibration is needed.

Output tensor shape: (grid_size, grid_size, 5 + num_classes)
  channel 0        : objectness (sigmoid) -- is there a spot centered in this cell?
  channels 1-4      : box regression (sigmoid) -- tx, ty (center offset within
                      the cell, 0-1) and tw, th (box width/height as a
                      fraction of the full image, 0-1)
  channels 5..      : class probabilities (softmax over num_classes)
"""

import keras
from tensorflow.keras import layers
import tensorflow as tf

INPUT_SIZE = 256
GRID_SIZE = 16
NUM_CLASSES = 2
CLASS_NAMES = ["empty_spot", "occupied_spot"]


@keras.saving.register_keras_serializable(package="smartpark")
def _activate_head(raw):
    """Applies sigmoid to objectness+box channels and softmax to the class
    channels, then concatenates back into one (S, S, 5+C) tensor."""
    obj_and_box = tf.sigmoid(raw[..., :5])
    class_logits = raw[..., 5:]
    class_probs = tf.nn.softmax(class_logits, axis=-1)
    return tf.concat([obj_and_box, class_probs], axis=-1)


def build_detector(input_size=INPUT_SIZE, grid_size=GRID_SIZE, num_classes=NUM_CLASSES):
    assert input_size % grid_size == 0, "grid_size must evenly divide input_size"
    n_downsamples = 4
    assert grid_size == input_size // (2 ** n_downsamples), (
        "this backbone downsamples by exactly 16x (4 stride-2 pools); "
        "grid_size must equal input_size // 16"
    )

    inputs = keras.Input(shape=(input_size, input_size, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)

    for filters in (16, 32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
    # x is now (grid_size, grid_size, 128)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)

    raw_head = layers.Conv2D(5 + num_classes, 1, padding="same", name="raw_head")(x)
    outputs = layers.Lambda(_activate_head, name="activated_head")(raw_head)

    model = keras.Model(inputs, outputs, name="parking_spot_detector")
    return model


if __name__ == "__main__":
    m = build_detector()
    m.summary()
