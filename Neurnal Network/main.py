# ============================================================
#  Car Brand Classifier - Full Activity
#  Covers: MLP + CNN, 3 data splits, model size reduction,
#          all required graphs
#
#  INSTALL:
#  pip install tensorflow matplotlib scikit-learn numpy
#
#  FOLDER STRUCTURE:
#  Neurnal-Network/
#  └── train/
#      ├── toyota/
#      ├── bmw/
#      └── ... (one subfolder per brand)
# ============================================================

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# ── SETTINGS ─────────────────────────────────────────────────

DATA_DIR   = "Neurnal-Network/train"   # <-- your images folder
IMG_SIZE   = (64, 64)    # smaller size = faster training for MLP
BATCH_SIZE = 32
EPOCHS     = 20

# The 3 splits required by the activity
SPLITS = {
    "70/30": 0.30,
    "90/10": 0.10,
    "50/50": 0.50,
}

# ── STEP 1: LOAD ALL IMAGES INTO MEMORY ──────────────────────
# We load everything once, then split manually per experiment.

print("\n[Loading images ...]\n")

full_datagen = ImageDataGenerator(rescale=1.0 / 255)
full_gen = full_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

NUM_CLASSES  = len(full_gen.class_indices)
CLASS_NAMES  = list(full_gen.class_indices.keys())
print(f"[OK] {NUM_CLASSES} brands found: {CLASS_NAMES}\n")

# Load all batches into numpy arrays
X_all, y_all = [], []
for i in range(len(full_gen)):
    xb, yb = full_gen[i]
    X_all.append(xb)
    y_all.append(yb)

X_all = np.concatenate(X_all, axis=0)
y_all = np.concatenate(y_all, axis=0)

print(f"[OK] Total images loaded: {X_all.shape[0]}\n")

# Flat version for MLP (pixels as a 1D vector)
X_flat = X_all.reshape(X_all.shape[0], -1)

# ── STEP 2: MODEL BUILDERS ────────────────────────────────────

def build_mlp(input_dim, num_classes, size_factor=1.0):
    """
    MLP model. size_factor=1.0 is full size.
    size_factor=0.75 = 25% reduction, 0.5 = 50% reduction.
    """
    units = max(1, int(256 * size_factor))
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(units, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(max(1, units // 2), activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ], name=f"MLP_x{size_factor}")
    return model


def build_cnn(input_shape, num_classes, size_factor=1.0):
    """
    CNN model. size_factor controls filter counts.
    """
    f1 = max(1, int(32 * size_factor))
    f2 = max(1, int(64 * size_factor))
    f3 = max(1, int(128 * size_factor))
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(f1, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(f2, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(f3, (3, 3), activation="relu", padding="same"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(max(1, int(128 * size_factor)), activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ], name=f"CNN_x{size_factor}")
    return model


def count_params(model):
    return model.count_params()


def train_and_eval(model, X_train, y_train, X_test, y_test, epochs=EPOCHS):
    """Compile, train, and return (accuracy, inference_time_ms)."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=4, restore_best_weights=True
            )
        ],
    )
    _, acc = model.evaluate(X_test, y_test, verbose=0)

    # Measure inference time for 1 sample
    sample = X_test[:1]
    start = time.perf_counter()
    model.predict(sample, verbose=0)
    inf_time_ms = (time.perf_counter() - start) * 1000

    return round(acc * 100, 2), round(inf_time_ms, 3)

# ── STEP 3: RUN ALL EXPERIMENTS ──────────────────────────────
# For each split: train MLP and CNN at full, 75%, and 50% size.

size_factors = {"Full (100%)": 1.0, "75% size": 0.75, "50% size": 0.50}

# Results storage
results = {
    "MLP": {sf: {} for sf in size_factors},
    "CNN": {sf: {} for sf in size_factors},
}

for split_name, test_size in SPLITS.items():
    print(f"\n{'='*55}")
    print(f" Split: {split_name}  (test={int(test_size*100)}%)")
    print(f"{'='*55}")

    X_tr_flat, X_te_flat, y_tr, y_te = train_test_split(
        X_flat, y_all, test_size=test_size, random_state=42, stratify=y_all
    )
    X_tr_img,  X_te_img,  _,    _   = train_test_split(
        X_all,  y_all, test_size=test_size, random_state=42, stratify=y_all
    )

    for sf_name, sf_val in size_factors.items():

        # --- MLP ---
        print(f"  MLP  {sf_name} ...", end=" ", flush=True)
        mlp = build_mlp(X_flat.shape[1], NUM_CLASSES, sf_val)
        acc, inf = train_and_eval(mlp, X_tr_flat, y_tr, X_te_flat, y_te)
        params = count_params(mlp)
        results["MLP"][sf_name][split_name] = {
            "acc": acc, "inf_ms": inf, "params": params
        }
        print(f"acc={acc}%  inf={inf}ms  params={params:,}")

        # --- CNN ---
        print(f"  CNN  {sf_name} ...", end=" ", flush=True)
        cnn = build_cnn((*IMG_SIZE, 3), NUM_CLASSES, sf_val)
        acc, inf = train_and_eval(cnn, X_tr_img, y_tr, X_te_img, y_te)
        params = count_params(cnn)
        results["CNN"][sf_name][split_name] = {
            "acc": acc, "inf_ms": inf, "params": params
        }
        print(f"acc={acc}%  inf={inf}ms  params={params:,}")

# ── STEP 4: GENERATE ALL REQUIRED GRAPHS ─────────────────────

fig_dir = "activity_graphs"
os.makedirs(fig_dir, exist_ok=True)

split_labels  = list(SPLITS.keys())
sf_labels     = list(size_factors.keys())
model_types   = ["MLP", "CNN"]
colors        = ["#2196F3", "#FF5722", "#4CAF50"]

# ---------- Graph 1 & 2: Size vs Accuracy (one per model) ----
for mtype in model_types:
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, split_name in enumerate(split_labels):
        accs   = [results[mtype][sf][split_name]["acc"] for sf in sf_labels]
        params = [results[mtype][sf][split_name]["params"] for sf in sf_labels]
        ax.plot(params, accs, marker="o", label=f"Split {split_name}",
                color=colors[i])
        for p, a in zip(params, accs):
            ax.annotate(f"{a}%", (p, a), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)

    ax.set_title(f"{mtype} -- Model Size vs Accuracy")
    ax.set_xlabel("Number of Parameters")
    ax.set_ylabel("Test Accuracy (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(fig_dir, f"graph_size_vs_accuracy_{mtype}.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"[Saved] {path}")

# ---------- Graph 3 & 4: Training Size vs Accuracy -----------
for mtype in model_types:
    fig, ax = plt.subplots(figsize=(9, 5))
    # training size as % of total
    train_pcts = [100 - int(v * 100) for v in SPLITS.values()]

    for sf_name, color in zip(sf_labels, colors):
        accs = [results[mtype][sf_name][sp]["acc"] for sp in split_labels]
        ax.plot(train_pcts, accs, marker="s", label=sf_name, color=color)
        for pct, a in zip(train_pcts, accs):
            ax.annotate(f"{a}%", (pct, a), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)

    ax.set_title(f"{mtype} -- Training Size vs Accuracy")
    ax.set_xlabel("Training Set Size (%)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_xticks(train_pcts)
    ax.set_xticklabels([f"{p}%" for p in train_pcts])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(fig_dir, f"graph_trainsize_vs_accuracy_{mtype}.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"[Saved] {path}")

# ---------- Graph 5 & 6: Model Size vs Inference Time --------
for mtype in model_types:
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, split_name in enumerate(split_labels):
        params  = [results[mtype][sf][split_name]["params"] for sf in sf_labels]
        inf_tms = [results[mtype][sf][split_name]["inf_ms"] for sf in sf_labels]
        ax.plot(params, inf_tms, marker="^", label=f"Split {split_name}",
                color=colors[i])
        for p, t in zip(params, inf_tms):
            ax.annotate(f"{t}ms", (p, t), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)

    ax.set_title(f"{mtype} -- Model Size vs Inference Time")
    ax.set_xlabel("Number of Parameters")
    ax.set_ylabel("Inference Time (ms)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(fig_dir, f"graph_size_vs_inference_{mtype}.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"[Saved] {path}")

# ---------- Summary table ------------------------------------
print("\n" + "="*70)
print(f"{'Model':<6} {'Size':<16} {'Split':<8} {'Accuracy':>10} {'Inf(ms)':>10} {'Params':>12}")
print("="*70)
for mtype in model_types:
    for sf_name in sf_labels:
        for split_name in split_labels:
            r = results[mtype][sf_name][split_name]
            print(f"{mtype:<6} {sf_name:<16} {split_name:<8} "
                  f"{r['acc']:>9.2f}% {r['inf_ms']:>9.3f}ms {r['params']:>12,}")
print("="*70)
print(f"\n[DONE] All graphs saved to '{fig_dir}/' folder.")