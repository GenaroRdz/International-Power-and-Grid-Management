# ============================================================
#  Car Brand Image Classifier — Transfer Learning (MobileNetV2)
#  Framework: TensorFlow / Keras
# ============================================================
#
#  EXPECTED FOLDER STRUCTURE:
#  dataset/
#  ├── train/
#  │   ├── toyota/       ← folder name = class label
#  │   ├── bmw/
#  │   ├── ford/
#  │   └── ...
#  └── val/
#      ├── toyota/
#      ├── bmw/
#      ├── ford/
#      └── ...
#
#  INSTALL DEPENDENCIES:
#  pip install tensorflow matplotlib
# ============================================================

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── 1. SETTINGS ─────────────────────────────────────────────

IMG_SIZE    = (224, 224)   # MobileNetV2 expects 224×224
BATCH_SIZE  = 32
EPOCHS_HEAD = 10           # Phase 1: train only the new top layers
EPOCHS_FINE = 10           # Phase 2: fine-tune the whole model
TRAIN_DIR   = "dataset/train"
VAL_DIR     = "dataset/val"
MODEL_PATH  = "car_brand_model.h5"

# ── 2. DATA LOADING & AUGMENTATION ──────────────────────────

# Augmentation helps the model generalise by randomly flipping,
# zooming, and shifting training images.
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,          # normalise pixels to [0, 1]
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

NUM_CLASSES = len(train_gen.class_indices)
print(f"\n✅ Detected {NUM_CLASSES} car brands: {list(train_gen.class_indices.keys())}\n")

# ── 3. BUILD THE MODEL ───────────────────────────────────────

# Load MobileNetV2 pre-trained on ImageNet, without its top layer
base_model = MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,          # remove the original classifier
    weights="imagenet",
)
base_model.trainable = False    # freeze the base during phase 1

# Add our own classification head on top
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),                # reduces overfitting
    layers.Dense(NUM_CLASSES, activation="softmax"),  # one output per brand
])

model.summary()

# ── 4. PHASE 1 — TRAIN THE HEAD ONLY ────────────────────────

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

print("\n🚀 Phase 1: Training classification head …\n")
history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_HEAD,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    ],
)

# ── 5. PHASE 2 — FINE-TUNE THE WHOLE MODEL ──────────────────

# Unfreeze the top layers of the base model for fine-tuning.
# We keep the early layers frozen because they detect generic
# features (edges, textures) that are useful for any image task.
base_model.trainable = True
for layer in base_model.layers[:-30]:   # freeze everything except last 30 layers
    layer.trainable = False

# Use a much smaller learning rate to avoid destroying the
# pre-trained weights.
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

print("\n🔧 Phase 2: Fine-tuning the model …\n")
history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FINE,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True),
    ],
)

# ── 6. PLOT TRAINING HISTORY ─────────────────────────────────

def plot_history(h1, h2):
    acc  = h1.history["accuracy"]      + h2.history["accuracy"]
    val  = h1.history["val_accuracy"]  + h2.history["val_accuracy"]
    loss = h1.history["loss"]          + h2.history["loss"]
    vloss= h1.history["val_loss"]      + h2.history["val_loss"]

    epochs = range(1, len(acc) + 1)
    split  = len(h1.history["accuracy"])   # where phase 2 starts

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, train_vals, val_vals, title in [
        (ax1, acc,  val,  "Accuracy"),
        (ax2, loss, vloss,"Loss"),
    ]:
        ax.plot(epochs, train_vals, label="Train")
        ax.plot(epochs, val_vals,   label="Validation")
        ax.axvline(split, color="gray", linestyle="--", label="Fine-tune start")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.show()
    print("📊 Training plot saved as training_history.png")

plot_history(history1, history2)

# ── 7. EVALUATE ON VALIDATION SET ────────────────────────────

loss, acc = model.evaluate(val_gen)
print(f"\n🏁 Final validation accuracy: {acc * 100:.2f}%")

# ── 8. PREDICT ON A SINGLE IMAGE ─────────────────────────────

import numpy as np
from tensorflow.keras.preprocessing import image as keras_image

def predict_car_brand(img_path: str):
    """Load one image and return the predicted car brand."""
    img = keras_image.load_img(img_path, target_size=IMG_SIZE)
    arr = keras_image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)   # add batch dimension

    preds      = model.predict(arr)[0]
    class_names = {v: k for k, v in train_gen.class_indices.items()}
    top_idx    = int(np.argmax(preds))
    confidence = preds[top_idx] * 100

    print(f"\n🔍 Predicted brand : {class_names[top_idx]}")
    print(f"   Confidence      : {confidence:.1f}%")

    # Show top 3 predictions
    top3 = np.argsort(preds)[::-1][:3]
    print("\n   Top 3 predictions:")
    for i in top3:
        print(f"   • {class_names[i]:<15} {preds[i]*100:.1f}%")

    return class_names[top_idx], confidence

# ── Example usage (comment out if not needed) ─────────────────
# predict_car_brand("my_car.jpg")