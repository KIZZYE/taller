# Taller 3 – Cartografía Automatizada (Electiva IV – Deep Computer Vision, USTA)

Segmentación panóptica de imágenes satelitales de Bucaramanga (4 clases: Área Caminable,
Área Construida, Zona Verde, Carretera) con dos arquitecturas:

* **Mask2Former** (Swin-Tiny, Hugging Face), fine-tuning desde COCO-panóptico.
* **EffPS-Lite**: implementación propia *inspirada en* EfficientPS (EfficientNet-B3 + FPN, rama semántica,
  rama de instancias con Mask R-CNN para edificios y fusión panóptica). No es el código oficial de EfficientPS,
  que exige CUDA y mmdetection antiguo.

## Estructura
```
data/        dataset de Kaggle (se descarga en el notebook)
models/      mask2former_model.py, effps_model.py, weights/ (pesos .pt del mejor modelo)
src/         config, data (dataset/aumentos/segmentos), metrics, trainer, ensemble (TTA/ensamble), panoptic (visualización)
notebooks/   01_entrenamiento_colab.ipynb
outputs/     curvas de pérdida, métricas (json), tabla comparativa, ejemplos
train.py     entrenamiento desde terminal
```
main.py      app web (Streamlit)

## 1. Preparación de datos
Dataset: `cetaamb/reto-1-cartografa-automatizada` (imágenes `recorte_N.png` + `ground_truth_recorte_N.npy` de 4 canales).
Con tu `kaggle.json` (Kaggle → Settings → API → Create New Token):
```
kaggle datasets download -d cetaamb/reto-1-cartografa-automatizada -p data/reto-1-cartografa-automatizada --unzip
```
Preprocesamiento: redimensionado a 256×256, división 70/15/15 (semilla 42, guardada en `outputs/splits.json`),
aumentos (flips, rot. 90°, brillo) solo en train. Edificios = instancias (componentes conexas); resto = *stuff*.

## 2. Cómo entrenar
Recomendado: abrir `notebooks/01_entrenamiento_colab.ipynb` en Google Colab con GPU T4 y ejecutar en orden.
Alternativa por terminal (con GPU):
```
pip install -r requirements.txt
python train.py --model both --epochs 30
```
Se guarda el mejor modelo (mayor mIoU en validación) en `models/weights/mask2former_best.pt` y `effps_best.pt`.

**Modelo final de EffPS-Lite:** se reentrenó a **512 px** (las imágenes miden 500×500) con aumentos fuertes (recorte con zoom,
giros, color) → `models/weights/effps_512_best.pt` y `outputs/metrics_effps_512.json`. Mask2Former se entrenó a 256 px.
Cada modelo procesa la imagen al tamaño con el que se entrenó (la app lo hace sola).

## 3. Reproducir métricas
`train.py` y el notebook evalúan en el conjunto de test y escriben `outputs/metrics_<modelo>.json`.
* Métricas: IoU, precisión, recall y F1 (acumuladas sobre todo el test, promedio de las 4 clases) y mAP
  (AP a nivel de píxel por clase, promediado). Mínimos del taller: IoU ≥ 0.7, mAP ≥ 0.5, F1 ≥ 0.8.
* Umbral de binarización: se reporta con umbral fijo 0.5 (`baseline_umbral_0.5` en el json) y con un **umbral por clase
  elegido en validación** (`src/trainer.py::tune_thresholds`); el de test nunca se usa para elegirlos.
* Experimentos adicionales (sección 10 del notebook; `src/ensemble.py` y `EffPSPeso` en `models/effps_model.py`): TTA (8 transformaciones),
  ensambles, peso extra en Área Caminable y 3 semillas. Se reportan aparte y se guardan en `outputs/metrics_mejoras.json`.
* Resultados finales (test): Mask2Former 256 px y EffPS-Lite 512 px; el mAP cumple el mínimo del taller, IoU y F1 no (ver informe).
* La semilla (42) y la división train/val/test quedan en `outputs/splits.json`.

## 4. App web (en el computador local)
1. Instalar Python 3.10–3.13 y, dentro de la carpeta del proyecto (idealmente en un entorno virtual):
   ```
   pip install -r requirements.txt
   ```
   (en un PC sin GPU NVIDIA, PyTorch normal funciona en CPU).
2. Copiar los archivos generados en Colab (`resultados_taller3.zip`):
   * `models/weights/mask2former_best.pt` y `models/weights/effps_512_best.pt`
   * `outputs/metrics_mask2former.json`, `outputs/metrics_effps_512.json`, `outputs/curvas_perdida.png`, `outputs/ejemplos_test.png`
3. Ejecutar:
   ```
   streamlit run main.py
   ```
   Se abre en el navegador (http://localhost:8501). Subir una imagen satelital (del dataset o una captura de
   Google Maps): se muestran la imagen original y la segmentación panóptica de **ambos modelos lado a lado**,
   la cantidad de edificios y el área por clase. Las pestañas *Métricas* y *Curvas de pérdida* muestran los
   resultados del test.

Nota: la primera carga de la app tarda unos segundos; la inferencia en CPU toma ~1–3 s por imagen y modelo.
