"""
train.py

Trains the Empty/Occupied CNN classifier on a folder of labeled spot crops.

Expected folder layout (this is also what PKLot/CNRPark-EXT loaders produce,
see data/README.md):

    <dataset_dir>/spot_crops/empty/*.jpg
    <dataset_dir>/spot_crops/occupied/*.jpg

Usage:
    python train.py --dataset ./synthetic_dataset --epochs 12 --out ./trained_model
"""

import argparse
import os
import sys
import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(__file__))
from models.cnn_model import build_cnn

IMG_SIZE = 64


def load_dataset(dataset_dir, img_size=IMG_SIZE, val_split=0.2, seed=42, batch_size=32):
    crops_dir = os.path.join(dataset_dir, "spot_crops")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        crops_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="binary",
        class_names=["empty", "occupied"],  # ensures empty=0, occupied=1
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        crops_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="binary",
        class_names=["empty", "occupied"],
    )
    return train_ds, val_ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="./synthetic_dataset",
                     help="Path to dataset dir containing spot_crops/empty and spot_crops/occupied")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out", default="./trained_model")
    ap.add_argument("--resume_from", default=None,
                     help="Path to an existing .keras model to continue training (for splitting "
                          "long training runs across multiple shorter calls)")
    ap.add_argument("--no_early_stopping", action="store_true",
                     help="Disable early stopping (useful when running exactly 1 epoch per call)")
    ap.add_argument("--steps_per_epoch", type=int, default=None,
                     help="Cap steps per call so runtime is predictable when splitting training "
                          "across multiple shorter invocations")
    ap.add_argument("--validation_steps", type=int, default=None)
    args = ap.parse_args()

    print(f"Loading dataset from {args.dataset} ...")
    train_ds, val_ds = load_dataset(args.dataset, batch_size=args.batch_size)

    # small dataset -> caching is fine and speeds up training a lot
    train_ds = train_ds.cache().shuffle(500).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    if args.resume_from:
        print(f"Resuming from {args.resume_from}")
        model = tf.keras.models.load_model(args.resume_from)
    else:
        model = build_cnn(input_size=IMG_SIZE)
    model.summary()

    callbacks = []
    if not args.no_early_stopping:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True)
        )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.validation_steps,
    )

    os.makedirs(args.out, exist_ok=True)
    model_path = os.path.join(args.out, "parking_spot_cnn.keras")
    model.save(model_path)
    print(f"\nSaved trained model to {model_path}")

    val_loss, val_acc = model.evaluate(val_ds)
    print(f"\nFinal validation accuracy: {val_acc*100:.2f}%")

    # save a small report
    with open(os.path.join(args.out, "training_report.txt"), "w") as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Epochs run: {len(history.history['loss'])}\n")
        f.write(f"Final val accuracy: {val_acc*100:.2f}%\n")
        f.write(f"Final val loss: {val_loss:.4f}\n")


if __name__ == "__main__":
    main()
