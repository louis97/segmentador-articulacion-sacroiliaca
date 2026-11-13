# Hueso experto en STIR (sacro e ilíacos), registrado desde Kaggle

Carpeta: `anotacion/hueso-experto/`. Contenido: este README, `veredictos-hueso-experto.json` (lo lee el pipeline) y `comparativas/` (92 cortes aceptados y 16 rechazados de ejemplo, en PNG).

> **Estado: propuesta sin verdad de referencia.** Toda lectura visual citada aquí fue hecha por un asistente de IA y **debe confirmarla el equipo**. La licencia del dataset de origen figura como "Unknown" y la consulta a la profesora sigue pendiente.

## 1. Qué es y para qué sirve

El dataset de Kaggle `krzysztofrzecki/bone-marrow-oedema-data` trae máscaras expertas de hueso (`BoneLabels`: sacro e ilíacos) **dibujadas sobre las T1**. Las registramos sobre la STIR (T1 → STIR) y juzgamos corte por corte cuáles caen sobre el hueso con fiabilidad suficiente: una **referencia experta de sacro e ilíacos en STIR**.

Usos previstos (a decidir por el equipo): medir MedSAM2 en los cortes aceptados y, si se decide, entrenar o evaluar un modelo de hueso. La **articulación se deriva del hueso** (interlínea y franja de ≤ 17 px entre sacro e ilíaco, criterio v0), por eso importan los cortes donde ambos son vecinos (el "núcleo").

Las máscaras registradas **no se incluyen en el proyecto** (solo PNG de revisión y el JSON). Para recuperarlas hay que descargar el dataset y repetir el registro.

## 2. Cómo se lee el JSON

`veredictos-hueso-experto.json` tiene una cabecera (`version`, `fecha`, `fuente`, `descripcion`, `regla`, `campo_recomendado`, `equivalencia_mendeley_kaggle`: SIJ_1..9 → STIR 4, 8, 9, 10, 11, 27, 33, 38, 39 y STIR_41 = SIJ_1 reescalado; `evidencia`, `unidades`, `excluidos_d_distinto`) y la lista `casos` (34 STIR de Kaggle con igual número de cortes en T1 y STIR). Campos por caso:

| Campo | Significado |
|---|---|
| `n`, `paciente`, `mendeley` | Número `STIR_n` de Kaggle; paciente (`n=4` y `n=41` son `P_SIJ_1`: **mismo paciente**); `SIJ_k` si es uno de nuestros 9, si no `null`. |
| `veredicto` | `SOLO_EN_CORTES` (≥ 3 cortes de núcleo contiguos), `USABLE` (hoy ninguno) o `NO_USABLE`. |
| `cortes_usables_nucleo` | **Cortes recomendados** (índices 0-based): sacro ≥ 20 % del área máxima del volumen, HD95 ≤ 3.5 px y lectura visual sobre el hueso. |
| `cortes_usables_cola`, `cortes_usables` | Cola: cortes contiguos al núcleo con poco sacro (< 20 %) a ≤ 10 px del ilíaco (solo 7: n=32 corte 11; n=40 cortes 0–5). `cortes_usables` = núcleo + cola. |
| `hd95_por_corte` | HD95 (px; 1 px = 0.6 mm), máximo entre las comparaciones válidas del caso. `null` = sin hueso; `9999.0` = hueso en una sola de las dos máscaras. |
| `comparaciones_validas`, `evidencia` | Número de comparaciones independientes válidas y nivel: `alta` (≥ 3), `media` (2), `baja` (≤ 1; implica `NO_USABLE`). |
| `juxtaarticular_por_corte` | Por corte: ¿cada fragmento de sacro está a ≤ 10 px del ilíaco? (condición de la cola). |
| `cortes_mal`, `cortes_dudosos`, `dudas` | Lectura visual (IA): cortes claramente mal o dudosos, y texto libre con salvedades ("Tarea" y "ronda" son etapas internas de la investigación). |

También existen `transformacion_elegida`, `comparaciones`, `segunda_impl_convergida`, `segunda_impl_diverge`, `D`, `T1`, `bytes` y `tarea`: metadatos del registro, sin uso en el pipeline.

**Convenciones.** Corte 0 = anterior (orden del archivo de la STIR); índices 0-based. Etiquetas: **1 y 3 = ilíacos, 2 = sacro, 0 = fondo**.

```python
import json
d = json.load(open("anotacion/hueso-experto/veredictos-hueso-experto.json"))
aceptados = [c for c in d["casos"] if c["veredicto"] != "NO_USABLE"]
nucleo = {c["n"]: c["cortes_usables_nucleo"] for c in aceptados}            
solo_mendeley = [c for c in aceptados if c["mendeley"] is not None]
# n=4 y n=41 son el mismo paciente: agrupar por c["paciente"] antes de repartir en train/val.
```

**Leer la STIR para superponer máscaras.** Archivo `STIR/STIR_{n}_400_400_{D}_2_.raw` del dataset de Kaggle: enteros de 2 bytes **big-endian** (`>u2`), forma `(D, 400, 400)`.

```python
import numpy as np
n, D = 27, 20                       # D = c["D"] del JSON
stir = np.fromfile(f"STIR/STIR_{n}_400_400_{D}_2_.raw", dtype=">u2").reshape(D, 400, 400)
```

## 3. Cómo llegamos a estas conclusiones

1. **Nuestros 9 volúmenes están en Kaggle**: idénticos byte a byte (SHA-256) a 9 de los 42 STIR. SIJ_1 aparece dos veces: STIR_4 (idéntico) y STIR_41 (mismas imágenes con intensidades ≈ ×4.91, otra T1 y otras etiquetas).
2. **Las BoneLabels están en la grilla de T1** (matriz 400, 560, 640 u 800 px), no en la de STIR (400×400); en 8 de los 42 casos T1 y STIR tienen distinto número de cortes. Hay que registrarlas.
3. **Registro T1 → STIR** (SimpleITK): información mutua, rígido y luego afín 3D; **la identidad es candidata** (se elige la de mayor información mutua, tolerancia 0.01). En los 34 casos: afín 22, identidad 8, rígido 4.
4. **Supuestos de geometría** (no verificados con metadatos): campo de visión de 240 mm en T1, mismos cortes y grosor de 3 mm que la STIR, STIR de 0.6×0.6×3 mm.
5. **Se excluyeron los 8 casos con D distinto** (Kaggle n = 1, 6, 8, 10, 11, 34, 37, 38; entre los nuestros, SIJ_2, 4, 5 y 8): sin correspondencia corte a corte la posición en z es incierta (en la prueba, n=8 quedó incierto).
6. **Regla de veredicto.** Sin verdad de referencia, la fiabilidad se mide por **consistencia entre registros** (HD95 por corte, px del plano STIR). Núcleo: sacro ≥ 20 % del área máxima, HD95 ≤ 3.5 px y lectura visual (IA) sobre el hueso, con ≥ 3 cortes contiguos. Cola: contigua al núcleo, HD95 ≤ 3.5 px, sacro < 20 % y cada fragmento de sacro a ≤ 10 px del ilíaco; se detiene en el primer corte que falla. Evidencia baja implica `NO_USABLE`.
7. **Defecto corregido en la ronda 1:** la segunda implementación reducía también z (pirámide isótropa), sesgaba la información mutua y generaba falsos `NO_USABLE`; se corrigió y se rehicieron los 34 casos.
8. **Resultado:** 0 `USABLE`, 16 `SOLO_EN_CORTES` y 18 `NO_USABLE`. Los 16 aceptados suman 92 cortes (85 de núcleo y 7 de cola), 87 cortes STIR distintos y 15 pacientes; evidencia 11 alta y 5 media (n = 4, 7, 25, 28, 41). El pipeline debe usar `cortes_usables_nucleo`.
9. **Evidencia bibliográfica: indirecta.** No hay estudio que compare localizar la articulación con y sin segmentación previa de sacro e ilion.
   - El código de Cracovia (`Segmentation/Step6_Regions.cpp`) **deriva la línea y la región articular de las etiquetas de hueso**, con una región de 17 px desde el contorno óseo: misma familia de regla que la nuestra, aunque no idéntica.
   - Ablaciones en tareas vecinas, ambas favorables y ninguna sobre localización: Du et al. (2026, TC) suben la exactitud de clasificación de 0.533 a 0.744 al segmentar el hueso pélvico y recortar; Lee et al. (2021, RM) obtienen 93.55 % con un recorte derivado de los huesos frente a 83.45 % con la imagen completa.
   - Riesgo: una interlínea derivada de dos bordes hereda el error de ambos.

## 4. Los 92 cortes aceptados: tabla de los 16 volúmenes

MV = máscara de edema `GroundTruth_MajorityVoting`: "sí", "vacía" (sin edema) o "no" (no existe). 11 de los 16 tienen MV con edema.

| n | Mendeley | Núcleo (cortes) | Cola (cortes) | Evidencia | MV | Carpeta en `comparativas/` |
|---|---|---|---|---|---|---|
| 4 | SIJ_1 | 6–10 | — | media | sí | `STIR_04-SIJ_1/` |
| 7 | — | 6–11 | — | media | sí | `STIR_07/` |
| 17 | — | 7–11 | — | alta | no | `STIR_17/` |
| 18 | — | 6–13 | — | alta | no | `STIR_18/` |
| 20 | — | 7–11 | — | alta | no | `STIR_20/` |
| 23 | — | 6–11 | — | alta | no | `STIR_23/` |
| 24 | — | 9–11 | — | alta | sí | `STIR_24/` |
| 25 | — | 8–13 | — | media | vacía | `STIR_25/` |
| 27 | SIJ_6 | 7–12 | — | alta | sí | `STIR_27-SIJ_6/` |
| 28 | — | 7–11 | — | media | sí | `STIR_28/` |
| 31 | — | 12–14 | — | alta | sí | `STIR_31/` |
| 32 | — | 6–10 | 11 | alta | sí | `STIR_32/` |
| 33 | SIJ_7 | 7–11 | — | alta | sí | `STIR_33-SIJ_7/` |
| 36 | — | 5–10 | — | alta | sí | `STIR_36/` |
| 40 | — | 6–11 | 0–5 | alta | sí | `STIR_40/` |
| 41 | SIJ_1 | 6–10 | — | media | sí | `STIR_41-SIJ_1/` |

Cada `comparativas/<carpeta>/` tiene un PNG por corte aceptado (`corte_07_nucleo.png`, `corte_11_cola.png`): STIR sola | STIR con la máscara registrada (rojo/naranja = ilíacos, cian = sacro), con el HD95 en el título. Además, un **rechazado** (`rechazado_corte_XX.png`, el motivo va en el título; el de n=31 es solo HD95): el corte con hueso más cercano al núcleo que no se aceptó.

**Nota n=31:** el corte 14 de núcleo es discutible (el contorno del sacro cae sobre masas laterales; lectura de IA) y el 11 se rechaza solo por HD95. Los `NO_USABLE` no tienen carpeta. Conviene mirar los PNG y confirmar o corregir la lectura.

## 5. Limitaciones

- **Sin verdad de referencia.** HD95 ≤ 3.5 px mide consistencia entre registros, no exactitud.
- **Las lecturas visuales son de un asistente de IA**; las debe confirmar el equipo.
- Geometría supuesta, no medida (§3).
- **Licencia "Unknown"** en Kaggle; la consulta a la profesora sigue pendiente.
- **n=4 y n=41 son el mismo paciente** (SIJ_1): deben ir siempre en el mismo conjunto.
- Umbrales (20 %, 3.5 px, 10 px) fijados por nosotros.

## 6. Referencias

- Rzecki, K. (2021). *Bone marrow oedema data* (Versión 1) [Conjunto de datos]. Kaggle. https://www.kaggle.com/krzysztofrzecki/bone-marrow-oedema-data
- Gut, D. (2022). *Knee and sacroiliac joints MRI scans* (Version 1) [Data set]. Mendeley Data. https://doi.org/10.17632/425fcppvkn.1
- Rzecki, K., et al. (s. f.). *bone-marrow-oedema-detection* [Repositorio de código; archivo `Segmentation/Step6_Regions.cpp`]. GitHub. https://github.com/krzysztofrzecki/bone-marrow-oedema-detection
- Du, Y.-k., Liu, R., Guo, H., Li, C., Chen, P., Qiu, H., Shi, D.-d., Cheng, J., Yan, J., & Li, Y.-s. (2026). Automated grading and diagnosis of sacroiliitis on CT images using a 3D convolutional neural network: A multicenter retrospective study. *Scientific Reports, 16*(1), 14099. https://doi.org/10.1038/s41598-026-44911-9
- Lee, K. H., Choi, S. T., Lee, G. Y., Ha, Y. J., & Choi, S. I. (2021). Method for diagnosing the bone marrow edema of sacroiliac joint in patients with axial spondyloarthritis using magnetic resonance image analysis based on deep learning. *Diagnostics, 11*(7), 1156. https://doi.org/10.3390/diagnostics11071156

El detalle completo de la investigación (scripts, figuras por caso y rondas de revisión) está en `.docs/fase2/comparacion-kaggle/` (carpeta local, no versionada).
