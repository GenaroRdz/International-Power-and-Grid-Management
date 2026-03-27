# =============================================================
#  ACTIVIDAD - Clasificación de Imágenes con MLP y CNN
#  Dataset : MNIST (dígitos del 0 al 9)
#  Pasos   : 1, 2, 3 y 4
# =============================================================
#
#  ESTRUCTURA DE ARCHIVOS QUE CREA ESTE SCRIPT:
#
#    proyecto/
#    ├── main.py
#    ├── graficas/
#    │   ├── mlp_size_vs_accuracy.png
#    │   ├── mlp_curvas_entrenamiento.png
#    │   ├── cnn_size_vs_accuracy.png
#    │   ├── cnn_curvas_entrenamiento.png
#    │   └── comparacion_mlp_vs_cnn.png
#    └── train/
#        └── mnist_train.npz
#
#  INSTALACIÓN:
#    pip install tensorflow matplotlib numpy
#
# =============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras # type: ignore

tf.random.set_seed(42)
np.random.seed(42)

os.makedirs("graficas", exist_ok=True)
os.makedirs("train",    exist_ok=True)

print("Carpetas listas: graficas/ y train/")


# ==============================================================
# PASO 1 — Cargar el dataset y dividirlo 70% train / 30% test
# ==============================================================
print("\n" + "="*55)
print("PASO 1: Cargando MNIST y dividiendo el dataset")
print("="*55)

RUTA_NPZ = os.path.join("train", "mnist_train.npz")

if os.path.exists(RUTA_NPZ):
    print("  Cargando desde archivo local:", RUTA_NPZ)
    datos  = np.load(RUTA_NPZ)
    x_todo = datos["imagenes"]
    y_todo = datos["etiquetas"]
else:
    print("  Descargando MNIST por primera vez (requiere internet)...")
    (x_tr_orig, y_tr_orig), (x_te_orig, y_te_orig) = keras.datasets.mnist.load_data()
    x_todo = np.concatenate([x_tr_orig, x_te_orig], axis=0)
    y_todo = np.concatenate([y_tr_orig, y_te_orig], axis=0)
    np.savez(RUTA_NPZ, imagenes=x_todo, etiquetas=y_todo)
    print("  Dataset guardado en:", RUTA_NPZ)

print(f"\n  Total de imágenes : {len(x_todo)}")
print(f"  Tamaño de cada img: {x_todo[0].shape}  (28×28 píxeles)")
print(f"  Clases            : 10  (dígitos del 0 al 9)")

# Normalizar píxeles [0,255] → [0.0, 1.0]
x_todo = x_todo.astype("float32") / 255.0

# ── Versión APLANADA para el MLP (28×28 → 784) ───────────────
x_todo_flat = x_todo.reshape(len(x_todo), -1)

# ── Versión 2D para la CNN (28×28×1) ─────────────────────────
# La CNN necesita la imagen como cuadrícula, no como lista plana.
# El "1" al final indica que es escala de grises (1 canal de color).
# Una imagen a color tendría 3 canales (RGB).
x_todo_cnn = x_todo[..., np.newaxis]   # forma: (70000, 28, 28, 1)

# ── Split 70 / 30 ─────────────────────────────────────────────
split = int(len(x_todo) * 0.70)   # 49,000 para entrenar

x_train_mlp = x_todo_flat[:split];   x_test_mlp = x_todo_flat[split:]
x_train_cnn = x_todo_cnn[:split];    x_test_cnn = x_todo_cnn[split:]
y_train     = y_todo[:split];        y_test     = y_todo[split:]

print(f"\n  Split 70/30 aplicado:")
print(f"    Entrenamiento : {len(y_train)} imágenes")
print(f"    Prueba        : {len(y_test)}  imágenes")


# ==============================================================
# FUNCIÓN COMPARTIDA — Compilar y entrenar cualquier modelo
# ==============================================================
def entrenar(model, x_tr, y_tr, x_te, y_te, epocas=30, batch=128):
    """
    Compila, entrena y evalúa el modelo.
    Funciona igual para MLP y CNN.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    parar_temprano = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True
    )

    historial = model.fit(
        x_tr, y_tr,
        epochs=epocas,
        batch_size=batch,
        validation_split=0.1,
        callbacks=[parar_temprano],
        verbose=1
    )

    _, accuracy = model.evaluate(x_te, y_te, verbose=0)
    parametros  = model.count_params()
    return historial, accuracy, parametros


# ==============================================================
# FUNCIÓN — Construir un MLP
# ==============================================================
def crear_mlp(neuronas_por_capa, nombre="MLP"):
    """
    MLP: red de capas Dense completamente conectadas.
    Recibe un vector plano de 784 valores (imagen aplanada).
    """
    model = keras.Sequential(name=nombre)
    model.add(keras.layers.Input(shape=(784,)))

    for n in neuronas_por_capa:
        model.add(keras.layers.Dense(n, activation="relu"))
        model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(10, activation="softmax"))
    return model


# ==============================================================
# FUNCIÓN — Construir una CNN
# ==============================================================
def crear_cnn(filtros_por_bloque, dense_units, nombre="CNN"):
    """
    CNN: red convolucional que detecta patrones espaciales en la imagen.

    Cada 'bloque' tiene:
      - Conv2D   : aplica filtros (detectores de bordes, curvas, etc.)
      - MaxPool  : reduce la imagen a la mitad (toma el valor máximo
                   de cada cuadrado 2×2), conservando lo más importante

    Después de los bloques convolucionales:
      - Flatten  : aplana la salida 2D a un vector 1D
      - Dense    : capa densa final para clasificar
      - Dropout  : regularización para evitar sobreajuste

    filtros_por_bloque : lista con filtros por cada bloque Conv.
                         Ejemplo: [32, 64]  → 2 bloques
    dense_units        : neuronas en la capa densa final
    """
    model = keras.Sequential(name=nombre)
    model.add(keras.layers.Input(shape=(28, 28, 1)))

    for filtros in filtros_por_bloque:
        # Conv2D: kernel_size=3 significa un filtro de 3×3 píxeles
        model.add(keras.layers.Conv2D(filtros, kernel_size=3,
                                      activation="relu", padding="same"))
        # MaxPooling2D reduce el mapa de características a la mitad
        model.add(keras.layers.MaxPooling2D(pool_size=2))

    # Aplanar para conectar con la capa Dense
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(dense_units, activation="relu"))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(10, activation="softmax"))
    return model


# ==============================================================
# PASO 2 — MLP Completo (100%)
# ==============================================================
print("\n" + "="*55)
print("PASO 2: Entrenando MLP — Modelo Completo (100%)")
print("="*55)

ARQ_MLP_BASE   = [512, 256, 128]

modelo_mlp_100 = crear_mlp(ARQ_MLP_BASE, nombre="MLP_100pct")
modelo_mlp_100.summary()
hist_mlp_100, acc_mlp_100, params_mlp_100 = entrenar(
    modelo_mlp_100, x_train_mlp, y_train, x_test_mlp, y_test
)
print(f"\n  Accuracy MLP 100% : {acc_mlp_100*100:.2f}%")
print(f"  Parámetros        : {params_mlp_100:,}")


# ==============================================================
# PASO 3 — MLP reducido al 75% y al 50%
# ==============================================================
print("\n" + "="*55)
print("PASO 3a: MLP con 75% del tamaño  (-25% de neuronas)")
print("="*55)

arq_mlp_75     = [int(n * 0.75) for n in ARQ_MLP_BASE]   # [384, 192, 96]
print(f"  Neuronas: {arq_mlp_75}")
modelo_mlp_75  = crear_mlp(arq_mlp_75, nombre="MLP_75pct")
hist_mlp_75, acc_mlp_75, params_mlp_75 = entrenar(
    modelo_mlp_75, x_train_mlp, y_train, x_test_mlp, y_test
)
print(f"\n  Accuracy MLP 75% : {acc_mlp_75*100:.2f}%")
print(f"  Parámetros       : {params_mlp_75:,}")

print("\n" + "="*55)
print("PASO 3b: MLP con 50% del tamaño  (-50% de neuronas)")
print("="*55)

arq_mlp_50     = [int(n * 0.50) for n in ARQ_MLP_BASE]   # [256, 128, 64]
print(f"  Neuronas: {arq_mlp_50}")
modelo_mlp_50  = crear_mlp(arq_mlp_50, nombre="MLP_50pct")
hist_mlp_50, acc_mlp_50, params_mlp_50 = entrenar(
    modelo_mlp_50, x_train_mlp, y_train, x_test_mlp, y_test
)
print(f"\n  Accuracy MLP 50% : {acc_mlp_50*100:.2f}%")
print(f"  Parámetros       : {params_mlp_50:,}")


# ==============================================================
# PASO 4 — CNN Completa (100%)
# ==============================================================
print("\n" + "="*55)
print("PASO 4: Entrenando CNN — Modelo Completo (100%)")
print("="*55)

# Arquitectura base: 2 bloques convolucionales + 1 capa densa
#   Bloque 1: 32 filtros de 3×3
#   Bloque 2: 64 filtros de 3×3
#   Capa densa: 128 neuronas
FILTROS_CNN_BASE = [32, 64]
DENSE_CNN_BASE   = 128

modelo_cnn_100 = crear_cnn(FILTROS_CNN_BASE, DENSE_CNN_BASE,
                            nombre="CNN_100pct")
modelo_cnn_100.summary()
hist_cnn_100, acc_cnn_100, params_cnn_100 = entrenar(
    modelo_cnn_100, x_train_cnn, y_train, x_test_cnn, y_test
)
print(f"\n  Accuracy CNN 100% : {acc_cnn_100*100:.2f}%")
print(f"  Parámetros        : {params_cnn_100:,}")


# ==============================================================
# PASO 4 — CNN reducida al 75% y al 50%  (repite paso 3)
# ==============================================================
print("\n" + "="*55)
print("PASO 4 (cont.): CNN con 75% del tamaño")
print("="*55)

filtros_cnn_75 = [int(f * 0.75) for f in FILTROS_CNN_BASE]  # [24, 48]
dense_cnn_75   = int(DENSE_CNN_BASE * 0.75)                  # 96
print(f"  Filtros: {filtros_cnn_75}  |  Dense: {dense_cnn_75}")
modelo_cnn_75  = crear_cnn(filtros_cnn_75, dense_cnn_75,
                            nombre="CNN_75pct")
hist_cnn_75, acc_cnn_75, params_cnn_75 = entrenar(
    modelo_cnn_75, x_train_cnn, y_train, x_test_cnn, y_test
)
print(f"\n  Accuracy CNN 75% : {acc_cnn_75*100:.2f}%")
print(f"  Parámetros       : {params_cnn_75:,}")

print("\n" + "="*55)
print("PASO 4 (cont.): CNN con 50% del tamaño")
print("="*55)

filtros_cnn_50 = [int(f * 0.50) for f in FILTROS_CNN_BASE]  # [16, 32]
dense_cnn_50   = int(DENSE_CNN_BASE * 0.50)                  # 64
print(f"  Filtros: {filtros_cnn_50}  |  Dense: {dense_cnn_50}")
modelo_cnn_50  = crear_cnn(filtros_cnn_50, dense_cnn_50,
                            nombre="CNN_50pct")
hist_cnn_50, acc_cnn_50, params_cnn_50 = entrenar(
    modelo_cnn_50, x_train_cnn, y_train, x_test_cnn, y_test
)
print(f"\n  Accuracy CNN 50% : {acc_cnn_50*100:.2f}%")
print(f"  Parámetros       : {params_cnn_50:,}")


# ==============================================================
# RESUMEN FINAL EN CONSOLA
# ==============================================================
print("\n" + "="*55)
print("RESUMEN COMPLETO — Tamaño vs Accuracy (split 70/30)")
print("="*55)
print(f"  {'Modelo':<22} {'Parámetros':>12}  {'Accuracy':>10}")
print(f"  {'-'*48}")
print(f"  {'MLP 100% (base)':<22} {params_mlp_100:>12,}  {acc_mlp_100*100:>9.2f}%")
print(f"  {'MLP 75%  (-25%)':<22} {params_mlp_75:>12,}  {acc_mlp_75*100:>9.2f}%")
print(f"  {'MLP 50%  (-50%)':<22} {params_mlp_50:>12,}  {acc_mlp_50*100:>9.2f}%")
print(f"  {'-'*48}")
print(f"  {'CNN 100% (base)':<22} {params_cnn_100:>12,}  {acc_cnn_100*100:>9.2f}%")
print(f"  {'CNN 75%  (-25%)':<22} {params_cnn_75:>12,}  {acc_cnn_75*100:>9.2f}%")
print(f"  {'CNN 50%  (-50%)':<22} {params_cnn_50:>12,}  {acc_cnn_50*100:>9.2f}%")


# ==============================================================
# PASO 5 — GRÁFICAS
# ==============================================================
etiquetas  = ["100%\n(base)", "75%\n(-25%)", "50%\n(-50%)"]

acc_mlp = [acc_mlp_100*100, acc_mlp_75*100, acc_mlp_50*100]
acc_cnn = [acc_cnn_100*100, acc_cnn_75*100, acc_cnn_50*100]
par_mlp = [params_mlp_100, params_mlp_75, params_mlp_50]
par_cnn = [params_cnn_100, params_cnn_75, params_cnn_50]

# ── Gráfica 1: MLP — Tamaño vs Accuracy ───────────────────────
fig, ejes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("MLP en MNIST  —  Split 70/30", fontsize=13, fontweight="bold")

ejes[0].plot(etiquetas, acc_mlp, marker="o", color="steelblue",
             linewidth=2.5, markersize=9)
ejes[0].axhline(y=95, color="red", linestyle="--", label="Meta 95%")
for lbl, a in zip(etiquetas, acc_mlp):
    ejes[0].annotate(f"{a:.2f}%", (lbl, a),
                     textcoords="offset points", xytext=(0,10),
                     ha="center", fontsize=11, fontweight="bold")
ejes[0].set_title("Tamaño vs Accuracy");  ejes[0].set_xlabel("Tamaño del modelo")
ejes[0].set_ylabel("Accuracy (%)");       ejes[0].set_ylim(85, 102)
ejes[0].legend();                         ejes[0].grid(True, alpha=0.3)

ejes[1].bar(etiquetas, par_mlp, color=["steelblue","darkorange","green"],
            width=0.5, edgecolor="black")
for i, p in enumerate(par_mlp):
    ejes[1].text(i, p+1000, f"{p:,}", ha="center", fontsize=10)
ejes[1].set_title("# Parámetros"); ejes[1].set_xlabel("Tamaño del modelo")
ejes[1].set_ylabel("Parámetros");  ejes[1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
r = os.path.join("graficas", "mlp_size_vs_accuracy.png")
plt.savefig(r, dpi=150, bbox_inches="tight");  plt.show()
print(f"\n  Guardada: {r}")

# ── Gráfica 2: MLP — Curvas de entrenamiento ──────────────────
fig2, ax = plt.subplots(figsize=(9, 5))
ax.plot(hist_mlp_100.history["val_accuracy"], label="MLP 100%", color="steelblue",  lw=2)
ax.plot(hist_mlp_75.history["val_accuracy"],  label="MLP 75%",  color="darkorange", lw=2)
ax.plot(hist_mlp_50.history["val_accuracy"],  label="MLP 50%",  color="green",      lw=2)
ax.axhline(y=0.95, color="red", linestyle="--", label="Meta 95%")
ax.set_title("MLP — Accuracy de Validación por Época")
ax.set_xlabel("Época");  ax.set_ylabel("Accuracy")
ax.legend();             ax.grid(True, alpha=0.3)
plt.tight_layout()
r = os.path.join("graficas", "mlp_curvas_entrenamiento.png")
plt.savefig(r, dpi=150, bbox_inches="tight");  plt.show()
print(f"  Guardada: {r}")

# ── Gráfica 3: CNN — Tamaño vs Accuracy ───────────────────────
fig3, ejes3 = plt.subplots(1, 2, figsize=(13, 5))
fig3.suptitle("CNN en MNIST  —  Split 70/30", fontsize=13, fontweight="bold")

ejes3[0].plot(etiquetas, acc_cnn, marker="o", color="darkorange",
              linewidth=2.5, markersize=9)
ejes3[0].axhline(y=95, color="red", linestyle="--", label="Meta 95%")
for lbl, a in zip(etiquetas, acc_cnn):
    ejes3[0].annotate(f"{a:.2f}%", (lbl, a),
                      textcoords="offset points", xytext=(0,10),
                      ha="center", fontsize=11, fontweight="bold")
ejes3[0].set_title("Tamaño vs Accuracy");   ejes3[0].set_xlabel("Tamaño del modelo")
ejes3[0].set_ylabel("Accuracy (%)");        ejes3[0].set_ylim(85, 102)
ejes3[0].legend();                          ejes3[0].grid(True, alpha=0.3)

ejes3[1].bar(etiquetas, par_cnn, color=["darkorange","steelblue","green"],
             width=0.5, edgecolor="black")
for i, p in enumerate(par_cnn):
    ejes3[1].text(i, p+500, f"{p:,}", ha="center", fontsize=10)
ejes3[1].set_title("# Parámetros"); ejes3[1].set_xlabel("Tamaño del modelo")
ejes3[1].set_ylabel("Parámetros");  ejes3[1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
r = os.path.join("graficas", "cnn_size_vs_accuracy.png")
plt.savefig(r, dpi=150, bbox_inches="tight");  plt.show()
print(f"  Guardada: {r}")

# ── Gráfica 4: CNN — Curvas de entrenamiento ──────────────────
fig4, ax4 = plt.subplots(figsize=(9, 5))
ax4.plot(hist_cnn_100.history["val_accuracy"], label="CNN 100%", color="darkorange", lw=2)
ax4.plot(hist_cnn_75.history["val_accuracy"],  label="CNN 75%",  color="steelblue",  lw=2)
ax4.plot(hist_cnn_50.history["val_accuracy"],  label="CNN 50%",  color="green",      lw=2)
ax4.axhline(y=0.95, color="red", linestyle="--", label="Meta 95%")
ax4.set_title("CNN — Accuracy de Validación por Época")
ax4.set_xlabel("Época");  ax4.set_ylabel("Accuracy")
ax4.legend();             ax4.grid(True, alpha=0.3)
plt.tight_layout()
r = os.path.join("graficas", "cnn_curvas_entrenamiento.png")
plt.savefig(r, dpi=150, bbox_inches="tight");  plt.show()
print(f"  Guardada: {r}")

# ── Gráfica 5: Comparación directa MLP vs CNN ─────────────────
fig5, ax5 = plt.subplots(figsize=(9, 5))
x = np.arange(len(etiquetas))
ancho = 0.35

barras_mlp = ax5.bar(x - ancho/2, acc_mlp, ancho,
                     label="MLP", color="steelblue", edgecolor="black")
barras_cnn = ax5.bar(x + ancho/2, acc_cnn, ancho,
                     label="CNN", color="darkorange", edgecolor="black")

# Etiquetas encima de cada barra
for barra in barras_mlp:
    ax5.text(barra.get_x() + barra.get_width()/2,
             barra.get_height() + 0.1,
             f"{barra.get_height():.2f}%", ha="center", fontsize=9)
for barra in barras_cnn:
    ax5.text(barra.get_x() + barra.get_width()/2,
             barra.get_height() + 0.1,
             f"{barra.get_height():.2f}%", ha="center", fontsize=9)

ax5.axhline(y=95, color="red", linestyle="--", linewidth=1.5, label="Meta 95%")
ax5.set_title("Comparación MLP vs CNN — Accuracy por Tamaño de Modelo",
              fontsize=11)
ax5.set_xticks(x);         ax5.set_xticklabels(etiquetas)
ax5.set_xlabel("Tamaño del modelo")
ax5.set_ylabel("Accuracy en prueba (%)")
ax5.set_ylim(85, 102);     ax5.legend();  ax5.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
r = os.path.join("graficas", "comparacion_mlp_vs_cnn.png")
plt.savefig(r, dpi=150, bbox_inches="tight");  plt.show()
print(f"  Guardada: {r}")

print("\n  Pasos 1, 2, 3 y 4 completados.")
print(f"    Dataset en  : train/mnist_train.npz")
print(f"    Gráficas en : graficas/")