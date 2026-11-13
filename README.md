# Segmentador de la articulación sacroilíaca

Preparación de volúmenes de RM STIR de la articulación sacroilíaca (iteración 1, paquete `sij_prep`) y evaluación de MedSAM2 sobre ellos (Fase 2, paquete `sij_medsam2`).

## Fase 2: evaluación de MedSAM2

Mide qué tan bien segmenta MedSAM2 el sacro y los ilíacos en RM STIR cuando recibe como pista la máscara experta de algunos cortes (o una caja), contra la máscara experta de los cortes que no recibió.

**Requisitos.** Una Mac con Apple Silicon (MPS; en CPU funciona pero es mucho más lento) y `uv sync`. Hace falta conexión la primera vez: el conjunto de Kaggle (unos 400 MB) y el checkpoint de MedSAM2 (156 MB, revisión fija de Hugging Face, con verificación de sha256) se descargan solos.

**Datos de partida (clon limpio).** `data/` no se versiona.
- Modo `kaggle`: no hace falta nada más; el conjunto se descarga solo a `data/external/kaggle-bmo/`.
- Modo `mendeley`: necesita además `data/raw/` (los 9 volúmenes de Mendeley). Se obtiene con `sij_prep.download.ensure_dataset(Path("data/raw"))` o ejecutando el notebook `notebooks/01_preparacion_imagenes.ipynb`.
- **El modo `mendeley` también necesita la descarga de Kaggle**, porque las máscaras de hueso (BoneLabels) solo vienen en ese conjunto; se descarga sola al correrlo.

Los parámetros de preprocesado (recorte de percentiles 0.5–99.5, N4, espaciado) son constantes de `sij_medsam2/datos.py`; no se necesita `data/processed/preprocessing_params.json` (si existe y difiere, se avisa).

**Cómo correr.** Abrir `notebooks/02_medsam2_evaluacion.ipynb` y elegir en la primera celda `DATASET = "kaggle"` (15 pacientes) o `"mendeley"` (3 pacientes, leídos de `data/raw/`). O desde la terminal:

```bash
uv run python -c "from sij_medsam2 import ejecutar_pipeline; ejecutar_pipeline('kaggle')"
```

**Resultados.** Quedan en `resultados/medsam2/<dataset>/`: `metricas_por_corte.csv`, `metricas_por_caso.csv`, `resumen.csv`, `excluidos.csv`, `configuracion.json` y las figuras de error. Las predicciones y las imágenes de ejemplo con RM van a `data/processed/medsam2/<dataset>/` y no se versionan (`data/` está ignorado, igual que las imágenes y máscaras de los conjuntos). Las predicciones guardadas se reutilizan (se invalidan si cambian las pistas); el notebook las vuelve a calcular con `REHACER = True`.

**Regenerar los resultados versionados.** Los CSV y JSON de `resultados/` se regeneran en la Mac (con MPS) con `ejecutar_pipeline(dataset, "mps")`, por ejemplo `ejecutar_pipeline("kaggle", "mps")`. Correr en CPU o sin las predicciones guardadas cambia `dispositivo_pedido` en `configuracion.json`.

**Notebooks sin salidas.** Los `.ipynb` se versionan sin salidas (el notebook 02 incrusta cortes de RM de Kaggle). `.gitattributes` aplica el filtro `nbstripout`; después de clonar, instalarlo una vez con `uv run nbstripout --install --attributes .gitattributes`.
