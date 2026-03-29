# =============================================================
#  ACTIVIDAD - Clasificación de Imágenes con MLP y CNN
#  Dataset : MNIST (dígitos del 0 al 9)
#  Pasos   : 1, 2, 3, 4, 5, 6, 7 y 8
# =============================================================
#
#  ESTRUCTURA DE ARCHIVOS:
#
#    proyecto/
#    ├── main.py
#    ├── graficas/
#    │   ├── split_70_30/
#    │   │   ├── mlp_size_vs_accuracy.png
#    │   │   ├── mlp_curvas_entrenamiento.png
#    │   │   ├── cnn_size_vs_accuracy.png
#    │   │   ├── cnn_curvas_entrenamiento.png
#    │   │   └── comparacion_mlp_vs_cnn.png
#    │   ├── split_90_10/   (mismas gráficas)
#    │   ├── split_50_50/   (mismas gráficas)
#    │   ├── training_size_vs_accuracy.png   ← paso 7
#    │   └── inferencia_tiempo.png           ← paso 8
#    └── train/
#        └── mnist_train.npz
#
#  INSTALACIÓN:
#    pip install tensorflow matplotlib numpy
#
# =============================================================

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras # type: ignore

tf.random.set_seed(42)
np.random.seed(42)

os.makedirs("train", exist_ok=True)


# ==============================================================
# PASO 1 — Cargar MNIST una sola vez
# ==============================================================
print("\n" + "="*55)
print("PASO 1: Cargando MNIST")
print("="*55)

RUTA_NPZ = os.path.join("train", "mnist_train.npz")

if os.path.exists(RUTA_NPZ):
    print("  Cargando desde archivo local:", RUTA_NPZ)
    datos  = np.load(RUTA_NPZ)
    x_todo = datos["imagenes"]
    y_todo = datos["etiquetas"]
else:
    print("  Descargando MNIST (requiere internet)...")
    (x_tr_o, y_tr_o), (x_te_o, y_te_o) = keras.datasets.mnist.load_data()
    x_todo = np.concatenate([x_tr_o, x_te_o], axis=0)
    y_todo = np.concatenate([y_tr_o, y_te_o], axis=0)
    np.savez(RUTA_NPZ, imagenes=x_todo, etiquetas=y_todo)
    print("  Dataset guardado en:", RUTA_NPZ)

print(f"  Total: {len(x_todo)} imágenes de 28×28 píxeles, 10 clases")

# Normalizar [0,255] → [0.0, 1.0]
x_todo = x_todo.astype("float32") / 255.0

# Dos versiones del dato:
#   - Plana  (70000, 784)    → para el MLP
#   - 2D     (70000, 28,28,1) → para la CNN
x_flat = x_todo.reshape(len(x_todo), -1)
x_cnn  = x_todo[..., np.newaxis]


# ==============================================================
# FUNCIONES — Construcción de modelos
# ==============================================================
def crear_mlp(neuronas, nombre="MLP"):
    """
    MLP: capas Dense completamente conectadas.
    Recibe imagen aplanada (vector de 784 valores).
    """
    m = keras.Sequential(name=nombre)
    m.add(keras.layers.Input(shape=(784,)))
    for n in neuronas:
        m.add(keras.layers.Dense(n, activation="relu"))
        m.add(keras.layers.Dropout(0.3))
    m.add(keras.layers.Dense(10, activation="softmax"))
    return m


def crear_cnn(filtros, dense_units, nombre="CNN"):
    """
    CNN: bloques Conv2D + MaxPooling para detectar patrones en la imagen.

    Cada bloque:
      Conv2D    → aplica filtros de 3×3 sobre la imagen buscando patrones
      MaxPool   → reduce la imagen a la mitad conservando lo más relevante

    Luego Flatten + Dense para la clasificación final.
    """
    m = keras.Sequential(name=nombre)
    m.add(keras.layers.Input(shape=(28, 28, 1)))
    for f in filtros:
        m.add(keras.layers.Conv2D(f, kernel_size=3, activation="relu", padding="same"))
        m.add(keras.layers.MaxPooling2D(pool_size=2))
    m.add(keras.layers.Flatten())
    m.add(keras.layers.Dense(dense_units, activation="relu"))
    m.add(keras.layers.Dropout(0.3))
    m.add(keras.layers.Dense(10, activation="softmax"))
    return m


# ==============================================================
# FUNCIÓN — Entrenar y evaluar (MLP o CNN)
# ==============================================================
def entrenar(model, x_tr, y_tr, x_te, y_te, epocas=30, batch=128):
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    parar = keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True
    )
    hist = model.fit(x_tr, y_tr, epochs=epocas, batch_size=batch,
                     validation_split=0.1, callbacks=[parar], verbose=1)
    _, acc = model.evaluate(x_te, y_te, verbose=0)
    return hist, acc, model.count_params()


# ==============================================================
# FUNCIÓN PRINCIPAL — Corre pasos 2,3,4,5 para un split dado
# Devuelve un diccionario con todos los resultados
# ==============================================================
def correr_experimento(porcentaje_train, x_flat, x_cnn, y_todo):
    """
    Entrena los 6 modelos (MLP 100/75/50% y CNN 100/75/50%)
    con el split indicado y guarda las gráficas en su carpeta.

    porcentaje_train : fracción de datos para entrenar (ej. 0.70)
    Retorna dict con accuracy y parámetros de cada modelo.
    """
    # ── Nombre de carpeta para las gráficas ───────────────────
    tren_pct = int(porcentaje_train * 100)
    test_pct = 100 - tren_pct
    carpeta  = os.path.join("graficas", f"split_{tren_pct}_{test_pct}")
    os.makedirs(carpeta, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  SPLIT  {tren_pct}% TRAIN  /  {test_pct}% TEST")
    print(f"{'='*55}")

    # ── Dividir el dataset ────────────────────────────────────
    split = int(len(y_todo) * porcentaje_train)

    x_tr_mlp = x_flat[:split];  x_te_mlp = x_flat[split:]
    x_tr_cnn = x_cnn[:split];   x_te_cnn = x_cnn[split:]
    y_tr     = y_todo[:split];  y_te     = y_todo[split:]

    print(f"  Entrenamiento: {len(y_tr):,} imágenes")
    print(f"  Prueba       : {len(y_te):,} imágenes")

    # ── Arquitecturas base ────────────────────────────────────
    ARQ_MLP    = [512, 256, 128]
    FILTROS    = [32, 64]
    DENSE_CNN  = 128

    # ── Entrenar los 6 modelos ────────────────────────────────
    print(f"\n  [MLP 100%] neuronas: {ARQ_MLP}")
    model_mlp_100 = crear_mlp(ARQ_MLP, "MLP_100")
    h_m100, a_m100, p_m100 = entrenar(model_mlp_100, x_tr_mlp, y_tr, x_te_mlp, y_te)

    arq75 = [int(n*0.75) for n in ARQ_MLP]
    print(f"\n  [MLP  75%] neuronas: {arq75}")
    model_mlp_75 = crear_mlp(arq75, "MLP_75")
    h_m75,  a_m75,  p_m75  = entrenar(model_mlp_75, x_tr_mlp, y_tr, x_te_mlp, y_te)

    arq50 = [int(n*0.50) for n in ARQ_MLP]
    print(f"\n  [MLP  50%] neuronas: {arq50}")
    model_mlp_50 = crear_mlp(arq50, "MLP_50")
    h_m50,  a_m50,  p_m50  = entrenar(model_mlp_50, x_tr_mlp, y_tr, x_te_mlp, y_te)

    print(f"\n  [CNN 100%] filtros: {FILTROS}  dense: {DENSE_CNN}")
    model_cnn_100 = crear_cnn(FILTROS, DENSE_CNN, "CNN_100")
    h_c100, a_c100, p_c100 = entrenar(model_cnn_100, x_tr_cnn, y_tr, x_te_cnn, y_te)

    f75 = [int(f*0.75) for f in FILTROS];  d75 = int(DENSE_CNN*0.75)
    print(f"\n  [CNN  75%] filtros: {f75}  dense: {d75}")
    model_cnn_75 = crear_cnn(f75, d75, "CNN_75")
    h_c75,  a_c75,  p_c75  = entrenar(model_cnn_75, x_tr_cnn, y_tr, x_te_cnn, y_te)

    f50 = [int(f*0.50) for f in FILTROS];  d50 = int(DENSE_CNN*0.50)
    print(f"\n  [CNN  50%] filtros: {f50}  dense: {d50}")
    model_cnn_50 = crear_cnn(f50, d50, "CNN_50")
    h_c50,  a_c50,  p_c50  = entrenar(model_cnn_50, x_tr_cnn, y_tr, x_te_cnn, y_te)

    # ── Resumen en consola ────────────────────────────────────
    print(f"\n  {'Modelo':<20} {'Params':>10}  {'Accuracy':>10}")
    print(f"  {'-'*44}")
    for nombre, a, p in [
        ("MLP 100%", a_m100, p_m100), ("MLP  75%", a_m75, p_m75),
        ("MLP  50%", a_m50, p_m50),   ("CNN 100%", a_c100, p_c100),
        ("CNN  75%", a_c75, p_c75),   ("CNN  50%", a_c50, p_c50)
    ]:
        print(f"  {nombre:<20} {p:>10,}  {a*100:>9.2f}%")

    # ── GRÁFICAS ──────────────────────────────────────────────
    etiq      = ["100%\n(base)", "75%\n(-25%)", "50%\n(-50%)"]
    acc_mlp   = [a_m100*100, a_m75*100, a_m50*100]
    acc_cnn   = [a_c100*100, a_c75*100, a_c50*100]
    par_mlp   = [p_m100, p_m75, p_m50]
    par_cnn   = [p_c100, p_c75, p_c50]
    titulo    = f"Split {tren_pct}% Train / {test_pct}% Test"

    def guardar(fig, nombre_archivo):
        ruta = os.path.join(carpeta, nombre_archivo)
        fig.savefig(ruta, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Guardada: {ruta}")

    # MLP: size vs accuracy chart
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"MLP — {titulo}", fontsize=13, fontweight="bold")
    ax[0].plot(etiq, acc_mlp, marker="o", color="steelblue", lw=2.5, ms=9)
    ax[0].axhline(95, color="red", ls="--", label="Target 95%")
    for e, a in zip(etiq, acc_mlp):
        ax[0].annotate(f"{a:.2f}%", (e, a), textcoords="offset points",
                       xytext=(0,10), ha="center", fontsize=11, fontweight="bold")
    ax[0].set(title="Model Size vs Accuracy", xlabel="Model size",
              ylabel="Accuracy (%)", ylim=(75, 102))
    ax[0].legend(); ax[0].grid(True, alpha=0.3)
    ax[1].bar(etiq, par_mlp, color=["steelblue","darkorange","green"],
              width=0.5, edgecolor="black")
    for i, p in enumerate(par_mlp):
        ax[1].text(i, p+1000, f"{p:,}", ha="center", fontsize=10)
    ax[1].set(title="# Parameters", xlabel="Model size", ylabel="Parameters")
    ax[1].grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); guardar(fig, "mlp_size_vs_accuracy.png")

    # MLP training curves
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.plot(h_m100.history["val_accuracy"], label="MLP 100%", color="steelblue", lw=2)
    ax2.plot(h_m75.history["val_accuracy"],  label="MLP 75%",  color="darkorange", lw=2)
    ax2.plot(h_m50.history["val_accuracy"],  label="MLP 50%",  color="green",      lw=2)
    ax2.axhline(0.95, color="red", ls="--", label="Target 95%")
    ax2.set(title=f"MLP — Training Curves ({titulo})",
            xlabel="Epoch", ylabel="Accuracy")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout(); guardar(fig2, "mlp_curvas_entrenamiento.png")

    # CNN: size vs accuracy chart
    fig3, ax3 = plt.subplots(1, 2, figsize=(13, 5))
    fig3.suptitle(f"CNN — {titulo}", fontsize=13, fontweight="bold")
    ax3[0].plot(etiq, acc_cnn, marker="o", color="darkorange", lw=2.5, ms=9)
    ax3[0].axhline(95, color="red", ls="--", label="Target 95%")
    for e, a in zip(etiq, acc_cnn):
        ax3[0].annotate(f"{a:.2f}%", (e, a), textcoords="offset points",
                        xytext=(0,10), ha="center", fontsize=11, fontweight="bold")
    ax3[0].set(title="Model Size vs Accuracy", xlabel="Model size",
               ylabel="Accuracy (%)", ylim=(75, 102))
    ax3[0].legend(); ax3[0].grid(True, alpha=0.3)
    ax3[1].bar(etiq, par_cnn, color=["darkorange","steelblue","green"],
               width=0.5, edgecolor="black")
    for i, p in enumerate(par_cnn):
        ax3[1].text(i, p+500, f"{p:,}", ha="center", fontsize=10)
    ax3[1].set(title="# Parameters", xlabel="Model size", ylabel="Parameters")
    ax3[1].grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); guardar(fig3, "cnn_size_vs_accuracy.png")

    # CNN training curves
    fig4, ax4 = plt.subplots(figsize=(9, 5))
    ax4.plot(h_c100.history["val_accuracy"], label="CNN 100%", color="darkorange", lw=2)
    ax4.plot(h_c75.history["val_accuracy"],  label="CNN 75%",  color="steelblue",  lw=2)
    ax4.plot(h_c50.history["val_accuracy"],  label="CNN 50%",  color="green",      lw=2)
    ax4.axhline(0.95, color="red", ls="--", label="Target 95%")
    ax4.set(title=f"CNN — Training Curves ({titulo})",
            xlabel="Epoch", ylabel="Accuracy")
    ax4.legend(); ax4.grid(True, alpha=0.3)
    plt.tight_layout(); guardar(fig4, "cnn_curvas_entrenamiento.png")

    # MLP vs CNN comparison
    fig5, ax5 = plt.subplots(figsize=(9, 5))
    x = np.arange(len(etiq)); w = 0.35
    b1 = ax5.bar(x - w/2, acc_mlp, w, label="MLP", color="steelblue",  edgecolor="black")
    b2 = ax5.bar(x + w/2, acc_cnn, w, label="CNN", color="darkorange", edgecolor="black")
    for b in list(b1) + list(b2):
        ax5.text(b.get_x()+b.get_width()/2, b.get_height()+0.1,
                 f"{b.get_height():.2f}%", ha="center", fontsize=9)
    ax5.axhline(95, color="red", ls="--", lw=1.5, label="Target 95%")
    ax5.set(title=f"MLP vs CNN Comparison — {titulo}",
            ylabel="Accuracy (%)", ylim=(75, 102))
    ax5.set_xticks(x); ax5.set_xticklabels(etiq)
    ax5.legend(); ax5.grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); guardar(fig5, "comparacion_mlp_vs_cnn.png")

    # ── Devolver resultados para pasos 7 y 8 ─────────────────
    # También devolvemos los modelos del primer split (70/30)
    # para medir el tiempo de inferencia en el paso 8
    return {
        "split"    : porcentaje_train,
        "acc_mlp"  : acc_mlp,
        "acc_cnn"  : acc_cnn,
        "par_mlp"  : par_mlp,
        "par_cnn"  : par_cnn,
        # modelos guardados solo si es el primer split (70/30)
        "modelos"  : [
            ("MLP 100%", model_mlp_100, "mlp"),
            ("MLP  75%", model_mlp_75,  "mlp"),
            ("MLP  50%", model_mlp_50,  "mlp"),
            ("CNN 100%", model_cnn_100, "cnn"),
            ("CNN  75%", model_cnn_75,  "cnn"),
            ("CNN  50%", model_cnn_50,  "cnn"),
        ] if porcentaje_train == 0.70 else []
    }


# ==============================================================
# PASOS 2,3,4,5  →  split 70/30
# PASO 6.1       →  split 90/10
# PASO 6.2       →  split 50/50
# ==============================================================
resultados = []

for split_ratio in [0.70, 0.90, 0.50]:
    res = correr_experimento(split_ratio, x_flat, x_cnn, y_todo)
    resultados.append(res)


# ==============================================================
# PASO 7 — Gráfica: Tamaño del Training vs Accuracy
# ==============================================================
print("\n" + "="*55)
print("PASO 7: Gráfica Training Size vs Accuracy")
print("="*55)

os.makedirs("graficas", exist_ok=True)

# Etiquetas del eje X: los tres splits
splits_label = ["70%\n(49,000)", "90%\n(63,000)", "50%\n(35,000)"]

# Colores y estilos para cada tamaño de modelo
configs = [
    ("MLP 100%", "acc_mlp", 0, "steelblue",   "o",  "-"),
    ("MLP  75%", "acc_mlp", 1, "steelblue",   "s", "--"),
    ("MLP  50%", "acc_mlp", 2, "steelblue",   "^", ":"),
    ("CNN 100%", "acc_cnn", 0, "darkorange",  "o",  "-"),
    ("CNN  75%", "acc_cnn", 1, "darkorange",  "s", "--"),
    ("CNN  50%", "acc_cnn", 2, "darkorange",  "^", ":"),
]

fig6, ax6 = plt.subplots(figsize=(11, 6))

for nombre, clave, idx, color, marker, ls in configs:
    # Recogemos la accuracy de ese modelo en cada split
    valores = [r[clave][idx] for r in resultados]
    ax6.plot(splits_label, valores, marker=marker, color=color,
             linestyle=ls, linewidth=2, markersize=8, label=nombre)

ax6.axhline(y=95, color="red", linestyle="--", linewidth=1.5, label="Target 95%")
ax6.set_title("Training Size vs Accuracy  (MLP and CNN)", fontsize=13, fontweight="bold")
ax6.set_xlabel("Percentage of data used for training", fontsize=11)
ax6.set_ylabel("Test Accuracy (%)", fontsize=11)
ax6.set_ylim(75, 102)
ax6.legend(loc="lower right", fontsize=9)
ax6.grid(True, alpha=0.3)
plt.tight_layout()

ruta7 = os.path.join("graficas", "training_size_vs_accuracy.png")
plt.savefig(ruta7, dpi=150, bbox_inches="tight")
plt.show()
print(f"  Guardada: {ruta7}")

# ==============================================================
# RESUMEN FINAL
# ==============================================================
print("\n" + "="*55)
print("RESUMEN COMPLETO")
print("="*55)
etiq_mod = ["100%", "75%", "50%"]
for r in resultados:
    t = int(r["split"]*100); te = 100-t
    print(f"\n  Split {t}/{te}:")
    print(f"    {'Modelo':<12}  {'Accuracy':>10}")
    for i, e in enumerate(etiq_mod):
        print(f"    MLP {e:<6}  {r['acc_mlp'][i]:>9.2f}%")
    for i, e in enumerate(etiq_mod):
        print(f"    CNN {e:<6}  {r['acc_cnn'][i]:>9.2f}%")

# ==============================================================
# PASO 8 — Tiempo de inferencia por modelo
# ==============================================================
# La inferencia es el proceso de darle UNA imagen al modelo ya
# entrenado y obtener su predicción.  Queremos saber cuánto
# tiempo tarda cada modelo en responder para comparar velocidad
# vs tamaño vs accuracy.
#
# Estrategia:
#   - Tomamos UNA imagen del set de prueba
#   - La pasamos 200 veces por el modelo y promediamos el tiempo
#   - Repetir 200 veces reduce el ruido de medición del sistema
# ==============================================================
print("\n" + "="*55)
print("PASO 8: Tiempo de inferencia por modelo")
print("="*55)

# Recuperamos los modelos del split 70/30 (primer experimento)
modelos_70 = resultados[0]["modelos"]

# Una sola imagen de prueba
# MLP  necesita forma (1, 784)
# CNN  necesita forma (1, 28, 28, 1)
img_mlp = x_flat[0:1]       # shape (1, 784)
img_cnn = x_cnn[0:1]        # shape (1, 28, 28, 1)

REPETICIONES = 200           # cuántas veces repetimos la inferencia

nombres_inf  = []
tiempos_inf  = []
params_inf   = []

for nombre, modelo, tipo in modelos_70:
    img = img_mlp if tipo == "mlp" else img_cnn

    # Llamada de "calentamiento": la primera llamada siempre es
    # más lenta porque TensorFlow inicializa buffers internos.
    # La descartamos para no sesgar el promedio.
    modelo.predict(img, verbose=0)

    # Medimos el tiempo de REPETICIONES inferencias
    inicio = time.perf_counter()
    for _ in range(REPETICIONES):
        modelo.predict(img, verbose=0)
    fin = time.perf_counter()

    # Tiempo promedio por inferencia en milisegundos
    tiempo_ms = (fin - inicio) / REPETICIONES * 1000

    nombres_inf.append(nombre)
    tiempos_inf.append(tiempo_ms)
    params_inf.append(modelo.count_params())

    print(f"  {nombre:<12}  {modelo.count_params():>10,} params  →  {tiempo_ms:.3f} ms / inferencia")

# ── Gráfica: Tamaño del modelo (parámetros) vs Tiempo ─────────
colores = ["steelblue","steelblue","steelblue",
           "darkorange","darkorange","darkorange"]
marcadores = ["o","s","^","o","s","^"]

fig8, ejes8 = plt.subplots(1, 2, figsize=(14, 5))
fig8.suptitle("Model Size vs Inference Time",
              fontsize=13, fontweight="bold")

# Left panel: scatter plot (parameters vs time)
for i, (nom, params, t) in enumerate(zip(nombres_inf, params_inf, tiempos_inf)):
    ejes8[0].scatter(params, t, color=colores[i], marker=marcadores[i],
                     s=120, zorder=5, label=nom)
    ejes8[0].annotate(f"{t:.3f} ms", (params, t),
                      textcoords="offset points", xytext=(5, 5), fontsize=8)

# Trend lines for MLP and CNN separately
ejes8[0].plot(params_inf[:3], tiempos_inf[:3], color="steelblue",
              linestyle="--", linewidth=1.5, alpha=0.6, label="_mlp_line")
ejes8[0].plot(params_inf[3:], tiempos_inf[3:], color="darkorange",
              linestyle="--", linewidth=1.5, alpha=0.6, label="_cnn_line")

ejes8[0].set_title("# Parameters vs Inference Time")
ejes8[0].set_xlabel("Number of parameters")
ejes8[0].set_ylabel("Inference time (ms)")
ejes8[0].legend(fontsize=8, loc="upper left")
ejes8[0].grid(True, alpha=0.3)

# Right panel: horizontal bars comparing all 6 models
colores_barra = colores
ejes8[1].barh(nombres_inf, tiempos_inf, color=colores_barra,
              edgecolor="black", linewidth=0.8)
for i, t in enumerate(tiempos_inf):
    ejes8[1].text(t + 0.01, i, f"{t:.3f} ms", va="center", fontsize=9)
ejes8[1].set_title("Direct Comparison — Inference Time per Model")
ejes8[1].set_xlabel("Inference time (ms)")
ejes8[1].grid(True, alpha=0.3, axis="x")
ejes8[1].invert_yaxis()   # MLP 100% on top

plt.tight_layout()
ruta8 = os.path.join("graficas", "inferencia_tiempo.png")
plt.savefig(ruta8, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n  Guardada: {ruta8}")

print("\nPasos 1–8 completados.")
print("    Gráficas en: graficas/")