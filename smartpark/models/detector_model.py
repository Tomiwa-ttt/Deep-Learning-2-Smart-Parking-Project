"""
detector_model.py

A small single-shot object detector (YOLO-style grid detector, trained from
scratch) that finds and classifies parking spots directly in a full,
uncropped lot photo -- two classes, "empty_spot" and "occupied_spot".

Unlike models/cnn_model.py (which classifies a pre-cropped, pre-calibrated
spot patch), this model takes the whole image and outputs a grid of
(objectness, box, class) predictions, so no manual per-camera spot
calibration is needed.

Each grid cell predicts NUM_BOXES_PER_CELL candidate boxes, not just one.
An earlier one-box-per-cell version dropped 28.2% of real PKLot ground-truth
boxes to grid-cell collisions (real lots average ~57 spots/image, denser
than anywhere in training) -- capping recall well below what the model could
otherwise achieve. Measuring collision rate directly at several (grid_size,
boxes_per_cell) combinations on the real data showed 3 boxes/cell at the
*same* 16x16 grid resolution eliminates collisions entirely (28.25% -> 0.00%)
without the added compute of a finer grid.

Output tensor shape: (grid_size, grid_size, NUM_BOXES_PER_CELL * (5 + num_classes))
  -- logically (grid_size, grid_size, NUM_BOXES_PER_CELL, 5 + num_classes):
  channel 0        : objectness (sigmoid) -- is there a spot centered here?
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
NUM_BOXES_PER_CELL = 3
CLASS_NAMES = ["empty_spot", "occupied_spot"]


@keras.saving.register_keras_serializable(package="smartpark")
def _activate_head(raw):
    """Applies sigmoid to objectness+box channels and softmax to the class
    channels, independently per box slot (uses the module-level
    NUM_BOXES_PER_CELL/NUM_CLASSES constants directly rather than closure
    args, so this stays a plain serializable top-level function), then
    reshapes back to the flat (S, S, boxes_per_cell*(5+C)) tensor the model
    actually outputs."""
    box_size = 5 + NUM_CLASSES
    shape = tf.shape(raw)
    reshaped = tf.reshape(raw, tf.concat([shape[:-1], [NUM_BOXES_PER_CELL, box_size]], axis=0))
    obj_and_box = tf.sigmoid(reshaped[..., :5])
    class_probs = tf.nn.softmax(reshaped[..., 5:], axis=-1)
    activated = tf.concat([obj_and_box, class_probs], axis=-1)
    return tf.reshape(activated, shape)


def build_detector(input_size=INPUT_SIZE, grid_size=GRID_SIZE, num_classes=NUM_CLASSES,
                    boxes_per_cell=NUM_BOXES_PER_CELL):
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

    raw_head = layers.Conv2D(boxes_per_cell * (5 + num_classes), 1, padding="same", name="raw_head")(x)
    outputs = layers.Lambda(_activate_head, name="activated_head")(raw_head)

    model = keras.Model(inputs, outputs, name="parking_spot_detector")
    return model


if __name__ == "__main__":
    m = build_detector()
    m.summary()
