"""App web (Streamlit): segmentación panóptica de imágenes satelitales con Mask2Former y EffPS-Lite.

Ejecutar en el computador local (no en Colab):
    streamlit run main.py
"""
import io
import json
import time

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from models.effps_model import EffPS
from models.mask2former_model import Mask2FormerPS
from src.config import CLASS_COLORS, CLASS_NAMES, OUTPUTS_DIR, WEIGHTS_DIR
from src.metrics import image_metrics
from src.panoptic import (class_overlay, colorize_panoptic, draw_building_boxes,
                          segments_summary)

MODELS = {  # nombre en pantalla -> (clase, archivo de pesos, tag de métricas, tamaño de entrada del entrenamiento)
    "Mask2Former": (Mask2FormerPS, WEIGHTS_DIR / "mask2former_best.pt", "mask2former", 256),
    "EffPS-Lite": (EffPS, WEIGHTS_DIR / "effps_512_best.pt", "effps_512", 512),
}

st.set_page_config(page_title="Cartografía Automatizada", page_icon="🛰️", layout="wide")


# ----------------------------------------------------------------------------
# Carga de modelos y métricas (se hace una sola vez)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelos…")
def load_models():
    loaded, missing = {}, []
    for name, (cls, path, _, _) in MODELS.items():
        if path.exists():
            loaded[name] = cls.load(path, "cpu")
        else:
            missing.append(f"{name} ({path.name})")
    return loaded, missing


@st.cache_data
def load_metrics():
    out = {}
    for name, (_, _, tag, _) in MODELS.items():
        f = OUTPUTS_DIR / f"metrics_{tag}.json"
        if f.exists():
            out[name] = json.loads(f.read_text(encoding="utf-8"))
    return out


def prepare_image(img: Image.Image, size: int):
    """RGB -> redimensionada a size×size (como en el entrenamiento)."""
    rgb = np.array(img.convert("RGB").resize((size, size), Image.BILINEAR))
    x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255)[None]
    return rgb, x


def run_inference(model, x, score_thr):
    t0 = time.time()
    probs, pans = model.predict(x, panoptic=True, score_thr=score_thr)
    return pans[0], probs[0], time.time() - t0


def load_gt(file, size):
    """ground_truth_*.npy (H,W,4) -> tensor (4,size,size) binario."""
    gt = np.load(io.BytesIO(file.getvalue()))
    if gt.ndim != 3 or gt.shape[-1] != 4:
        raise ValueError(f"El .npy debe tener forma (alto, ancho, 4); se recibió {gt.shape}.")
    chans = [np.asarray(Image.fromarray(((gt[..., c] > 0) * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)) > 0
             for c in range(4)]
    return torch.from_numpy(np.stack(chans))


def test_metrics_df():
    rows = []
    for name, r in metrics.items():
        o = r["overall"]
        rows.append({"Modelo": name, "IoU": o["IoU"], "mAP": o["mAP"], "F1": o["F1"],
                     "Precisión": o["Precisión"], "Recall": o["Recall"]})
    return pd.DataFrame(rows).style.format({c: "{:.3f}" for c in ["IoU", "mAP", "F1", "Precisión", "Recall"]})


# ----------------------------------------------------------------------------
# Interfaz
# ----------------------------------------------------------------------------
st.title("🛰️ Cartografía Automatizada – Segmentación panóptica")
st.caption("Electiva IV – Deep Computer Vision · USTA Bucaramanga · Mask2Former vs. EffPS-Lite (inspirada en EfficientPS)")

models, missing = load_models()
metrics = load_metrics()
if missing:
    st.warning("No se encontraron los pesos de: " + ", ".join(missing) +
               ". Copia los archivos .pt a la carpeta `models/weights/` (ver README).")

with st.sidebar:
    st.header("Opciones")
    score_thr = st.slider("Umbral de confianza de segmentos", 0.1, 0.9, 0.5, 0.05,
                          help="Segmentos con menor confianza no se muestran.")
    alpha = st.slider("Opacidad de la segmentación", 0.2, 0.9, 0.55, 0.05)
    borders = st.checkbox("Dibujar bordes de segmentos", value=True)
    st.caption("Cada modelo procesa la imagen al tamaño con el que se entrenó: "
               + ", ".join(f"{n} {v[3]} px" for n, v in MODELS.items()) + ".")
    st.markdown("**Leyenda**")
    for i, n in enumerate(CLASS_NAMES):
        r, g, b = CLASS_COLORS[i]
        st.markdown(f"<span style='display:inline-block;width:14px;height:14px;background:rgb({r},{g},{b});"
                    f"border-radius:3px;margin-right:6px'></span>{n}", unsafe_allow_html=True)

tab_seg, tab_met, tab_curv = st.tabs(["🗺️ Segmentación", "📊 Métricas", "📈 Curvas de pérdida"])

# ---------------------------------------------------------------- segmentación
with tab_seg:
    c_up, c_gt = st.columns([3, 2])
    up = c_up.file_uploader("Sube una imagen satelital (una del dataset o una captura de Google Maps)",
                            type=["png", "jpg", "jpeg"])
    gt_up = c_gt.file_uploader("(Opcional) ground truth de esa imagen (.npy) para calcular las métricas sobre ella",
                               type=["npy"])
    if up is None:
        st.info("Sube una imagen para ver la segmentación de ambos modelos lado a lado.")
    elif not models:
        st.error("No hay modelos cargados.")
    else:
        img_in = Image.open(up)
        names = list(models)
        inputs = {n: prepare_image(img_in, MODELS[n][3]) for n in names}   # (rgb, tensor) al tamaño de cada modelo
        res = {}
        for name in names:
            with st.spinner(f"{name}…"):
                res[name] = run_inference(models[name], inputs[name][1], score_thr)   # (pan, probs, segundos)

        # 1) Comparación visual lado a lado
        st.subheader("Segmentación panóptica")
        cols = st.columns(1 + len(names))
        cols[0].image(np.array(img_in.convert("RGB")), caption="Imagen original", width="stretch")
        for col, name in zip(cols[1:], names):
            pan, _, dt = res[name]
            col.image(colorize_panoptic(inputs[name][0], pan, alpha=alpha, draw_borders=borders),
                      caption=f"{name} · {dt:.1f} s", width="stretch")

        # 2) Zonas detectadas (segmentos y área por clase)
        st.subheader("Zonas detectadas")
        cols = st.columns(len(names))
        for col, name in zip(cols, names):
            pan = res[name][0]
            n_build = sum(s["is_thing"] and s["label_id"] == 1 for s in pan["segments"])
            col.markdown(f"**{name}** — {n_build} instancias de área construida")
            col.dataframe(pd.DataFrame(segments_summary(pan)), hide_index=True, width="stretch")

        # 3) Métricas
        st.subheader("Métricas")
        if gt_up is not None:
            try:
                st.markdown("**En esta imagen** (contra el ground truth subido, con los umbrales por clase del entrenamiento; el mAP y el promedio "
                            "solo consideran las clases presentes en la imagen)")
                rows, per_cls = [], {}
                for name in names:
                    m = image_metrics(res[name][1], load_gt(gt_up, MODELS[name][3]), metrics.get(name, {}).get("thresholds", 0.5))
                    per_cls[name] = m["per_class"]
                    if m["overall"]:
                        o = m["overall"]
                        rows.append({"Modelo": name, "IoU": o["IoU"], "mAP": o["mAP"], "F1": o["F1"],
                                     "Precisión": o["Precisión"], "Recall": o["Recall"]})
                if rows:
                    st.dataframe(pd.DataFrame(rows).style.format({c: "{:.3f}" for c in ["IoU", "mAP", "F1", "Precisión", "Recall"]}),
                                 hide_index=True, width="stretch")
                cols = st.columns(len(names))
                for col, name in zip(cols, names):
                    col.caption(f"{name} – por clase")
                    d = pd.DataFrame({k: (v if v else {}) for k, v in per_cls[name].items()}).T
                    col.dataframe(d.style.format("{:.3f}", na_rep="—"), width="stretch")
            except Exception as e:  # archivo con forma inesperada
                st.error(f"No se pudo usar el ground truth: {e}")
        else:
            st.caption("Sube el `ground_truth_*.npy` de la imagen (opcional) para ver las métricas sobre esta imagen.")
        if metrics:
            st.markdown("**En el conjunto de test completo**")
            st.dataframe(test_metrics_df(), hide_index=True, width="stretch")

        # 4) Detección y segmentación de cada zona
        st.subheader("Detección y segmentación de cada zona")
        for name in names:
            pan, rgb = res[name][0], inputs[name][0]
            st.markdown(f"**{name}**")
            cols = st.columns(5)
            for c, col in enumerate(cols[:4]):
                img_c, pct = class_overlay(rgb, pan, c)
                col.image(img_c, caption=f"{CLASS_NAMES[c]} · {pct:.1f}% del área", width="stretch")
            boxes_img, n = draw_building_boxes(rgb, pan)
            cols[4].image(boxes_img, caption=f"Instancias de área construida: {n}", width="stretch")

# -------------------------------------------------------------------- métricas
with tab_met:
    if not metrics:
        st.info("Aún no hay métricas. Copia `outputs/metrics_*.json` (generados en el notebook) a la carpeta `outputs/`.")
    else:
        st.markdown("Resultados en el **conjunto de test** (umbral por clase ajustado en validación; "
                    "si el json no lo trae, 0.5).")
        st.dataframe(test_metrics_df(), hide_index=True, width="stretch")
        cols = st.columns(len(metrics))
        for col, (name, r) in zip(cols, metrics.items()):
            col.markdown(f"**{name} – por clase**")
            col.dataframe(pd.DataFrame(r["per_class"]).T.style.format("{:.3f}"), width="stretch")

# ---------------------------------------------------------------------- curvas
with tab_curv:
    f = OUTPUTS_DIR / "curvas_perdida.png"
    if f.exists():
        st.image(str(f), width="stretch")
    else:
        st.info("Copia `outputs/curvas_perdida.png` (generada en el notebook) a la carpeta `outputs/`.")
    f = OUTPUTS_DIR / "ejemplos_test.png"
    if f.exists():
        st.subheader("Ejemplos en test")
        st.image(str(f), width="stretch")
