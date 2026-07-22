"""
train_detector_mixed.py

Fine-tunes the synthetic-trained detector on a *mix* of synthetic and
augmented real PKLot data together, rather than real data alone.

Real-only fine-tuning (train_detector.py --resume_from ... --dataset
real_pklot_dataset) was tried first and caused catastrophic forgetting:
synthetic mAP collapsed from 98.71% to 6.50% after 25 epochs on real data
alone, a well-known transfer-learning failure mode (the model overwrites
what it learned about the original domain instead of extending it).
"Rehearsal" -- keeping some of the original-domain data in the fine-tuning
mix -- is the standard fix.

Usage:
    python train_detector_mixed.py --synthetic_dataset ./synthetic_dataset \
        --real_dataset ./real_pklot_dataset --epochs 25 \
        --resume_from ./trained_detector/spot_detector.keras --lr 1e-4 \
        --augment_crops 4 --out ./trained_detector_mixed
"""

import argparse
import numpy as np
import tensorflow as tf

from train_detector import load_dataset, build_crop_augmented_set
from detector_utils import detection_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic_dataset", default="./synthetic_dataset")
    ap.add_argument("--real_dataset", default="./real_pklot_dataset")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--val_split", type=float, default=0.15)
    ap.add_argument("--out", default="./trained_detector_mixed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume_from", default="./trained_detector/spot_detector.keras")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--augment_crops", type=int, default=4,
                     help="Random crops per real training image (0 to disable)")
    args = ap.parse_args()

    print(f"Loading synthetic dataset from {args.synthetic_dataset} ...")
    syn = load_dataset(args.synthetic_dataset)
    print(f"Loading real dataset from {args.real_dataset} ...")
    real = load_dataset(args.real_dataset)

    rng = np.random.RandomState(args.seed)

    def split(data):
        idx = rng.permutation(len(data["X"]))
        n_val = max(int(len(data["X"]) * args.val_split), 1)
        return idx[n_val:], idx[:n_val]

    syn_train_idx, syn_val_idx = split(syn)
    real_train_idx, real_val_idx = split(real)

    X_train_parts = [syn["X"][syn_train_idx], real["X"][real_train_idx]]
    Y_train_parts = [syn["Y"][syn_train_idx], real["Y"][real_train_idx]]

    if args.augment_crops > 0:
        print(f"Generating {args.augment_crops} random crop(s) per real training image ...")
        X_aug, Y_aug, n_generated = build_crop_augmented_set(
            real["image_paths"], real["raw_boxes"], real["raw_classes"], real_train_idx,
            n_crops=args.augment_crops, seed=args.seed,
        )
        if X_aug is not None:
            X_train_parts.append(X_aug)
            Y_train_parts.append(Y_aug)
        print(f"Added {n_generated} augmented real crops")

    X_train = np.concatenate(X_train_parts, axis=0)
    Y_train = np.concatenate(Y_train_parts, axis=0)
    X_val_syn, Y_val_syn = syn["X"][syn_val_idx], syn["Y"][syn_val_idx]
    X_val_real, Y_val_real = real["X"][real_val_idx], real["Y"][real_val_idx]
    # combined validation set just for the Keras training-loop callbacks;
    # the real per-domain numbers come from evaluate_detector.py afterward
    X_val = np.concatenate([X_val_syn, X_val_real], axis=0)
    Y_val = np.concatenate([Y_val_syn, Y_val_real], axis=0)

    print(f"Train: {len(X_train)} (synthetic {len(syn_train_idx)} + real {len(real_train_idx)} "
          f"+ augmented {len(X_train) - len(syn_train_idx) - len(real_train_idx)})")
    print(f"Val: {len(X_val)} (synthetic {len(X_val_syn)} + real {len(X_val_real)})")

    print(f"Resuming from {args.resume_from}")
    model = tf.keras.models.load_model(args.resume_from, compile=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss=detection_loss)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]

    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    import os
    os.makedirs(args.out, exist_ok=True)
    model_path = os.path.join(args.out, "spot_detector.keras")
    model.save(model_path)
    print(f"\nSaved trained detector to {model_path}")

    val_loss = model.evaluate(X_val, Y_val, verbose=0)
    with open(os.path.join(args.out, "training_report.txt"), "w") as f:
        f.write(f"Synthetic dataset: {args.synthetic_dataset}\n")
        f.write(f"Real dataset: {args.real_dataset}\n")
        f.write(f"Resumed from: {args.resume_from}\n")
        f.write(f"Augment crops per real training image: {args.augment_crops}\n")
        f.write(f"Train: {len(X_train)} (synthetic {len(syn_train_idx)} + real {len(real_train_idx)} "
                f"+ augmented {len(X_train) - len(syn_train_idx) - len(real_train_idx)})\n")
        f.write(f"Val (combined, for training callbacks only): {len(X_val)} "
                f"(synthetic {len(X_val_syn)} + real {len(X_val_real)})\n")
        f.write(f"Epochs run: {len(history.history['loss'])}\n")
        f.write(f"Final train loss: {history.history['loss'][-1]:.4f}\n")
        f.write(f"Final combined val loss: {val_loss:.4f}\n")
    print(f"Final combined val loss: {val_loss:.4f}")


if __name__ == "__main__":
    main()
