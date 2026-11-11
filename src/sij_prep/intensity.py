"""Intensidad: máscara corporal, corrección N4 y normalización a uint8."""
from __future__ import annotations

from typing import Literal

import numpy as np
import SimpleITK as sitk

from sij_prep.io import VOXEL_SPACING_XYZ_MM


def _to_sitk(arr: np.ndarray, spacing_xyz) -> sitk.Image:
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(tuple(float(s) for s in spacing_xyz))
    return img


def body_mask(
    vol: np.ndarray,
    method: Literal["otsu", "low_threshold"] = "low_threshold",
    low_threshold_fraction: float = 0.05,
) -> np.ndarray:
    """Máscara booleana (D,H,W) del cuerpo frente al fondo.

    `otsu` usa el umbral de Otsu de SimpleITK; `low_threshold` usa
    `vol > fracción * percentil 99`. En STIR la grasa suprimida es oscura y Otsu
    tiende a excluirla, por eso el umbral bajo es el valor por defecto. En ambos
    casos se rellenan huecos corte a corte y se conserva la mayor componente 3D.
    """
    vol = np.asarray(vol)
    if method == "otsu":
        img = sitk.GetImageFromArray(vol.astype(np.float32))
        m = sitk.OtsuThreshold(img, 0, 1)  # el cuerpo (clase alta) vale 1
        raw = sitk.GetArrayFromImage(m).astype(bool)
    elif method == "low_threshold":
        raw = vol > low_threshold_fraction * np.percentile(vol, 99)
    else:
        raise ValueError(f"Método de máscara desconocido: '{method}'. Use 'otsu' o 'low_threshold'.")

    filled = np.zeros(raw.shape, dtype=np.uint8)
    for k in range(raw.shape[0]):
        s = sitk.GetImageFromArray(raw[k].astype(np.uint8))
        filled[k] = sitk.GetArrayFromImage(sitk.BinaryFillhole(s))

    cc = sitk.ConnectedComponent(sitk.GetImageFromArray(filled))
    cc = sitk.RelabelComponent(cc, sortByObjectSize=True)  # etiqueta 1 = la mayor
    return sitk.GetArrayFromImage(cc) == 1


def n4_correct(
    vol: np.ndarray,
    mask: np.ndarray,
    spacing_xyz=VOXEL_SPACING_XYZ_MM,
    shrink_factor: int = 4,
    max_iterations: tuple[int, ...] = (50, 50, 50),
) -> tuple[np.ndarray, np.ndarray]:
    """Corrección de campo de sesgo N4; devuelve `(corregido, campo_de_sesgo)` float32.

    Se trabaja con `vol + 1` para evitar ceros en el fondo. El ajuste se hace
    sobre la imagen y la máscara reducidas y el campo se evalúa a resolución
    completa (`GetLogBiasFieldAsImage` + `exp`), por lo que `campo > 0`.
    La reducción es `[f, f, 1]`: en xy el vóxel es de 0.6 mm (f=4 da 2.4 mm,
    cercano al corte de 3 mm), y reducir z (solo 18-24 cortes) la dejaría con
    4-6 cortes, demasiado poco para ajustar el campo.
    Se usan 3 niveles de ajuste en vez de 4 porque con 4 el campo seguía la
    anatomía (zona caliente sobre el contenido intestinal brillante) y podía aplanar
    señal real; con 3 el campo es suave y el CV sigue bajando.
    """
    vol32 = np.asarray(vol, dtype=np.float32)
    img = _to_sitk(vol32 + 1.0, spacing_xyz)
    msk = _to_sitk(np.asarray(mask).astype(np.uint8), spacing_xyz)
    msk = sitk.Cast(msk, sitk.sitkUInt8)

    factors = [int(shrink_factor), int(shrink_factor), 1]
    small_img = sitk.Shrink(img, factors)
    small_msk = sitk.Shrink(msk, factors)

    n4 = sitk.N4BiasFieldCorrectionImageFilter()
    n4.SetMaximumNumberOfIterations(list(max_iterations))
    n4.Execute(small_img, small_msk)

    log_bias = n4.GetLogBiasFieldAsImage(img)
    bias = np.exp(sitk.GetArrayFromImage(log_bias).astype(np.float32)).astype(np.float32)
    corrected = ((vol32 + 1.0) / bias - 1.0).astype(np.float32)
    return corrected, bias


def coefficient_of_variation(vol: np.ndarray, mask: np.ndarray) -> float:
    """Coeficiente de variación `std/media` de los vóxeles dentro de la máscara."""
    v = np.asarray(vol, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    if v.size == 0:
        raise ValueError("La máscara está vacía: no hay vóxeles para calcular el coeficiente de variación.")
    return float(v.std() / v.mean())


def percentile_bounds(
    vol: np.ndarray, low_pct: float, high_pct: float, mask: np.ndarray | None = None
) -> tuple[float, float]:
    """Valores de los percentiles `low_pct` y `high_pct`; pase la máscara para excluir el fondo."""
    v = np.asarray(vol)
    if mask is not None:
        v = v[np.asarray(mask, dtype=bool)]
    if v.size == 0:
        raise ValueError("La máscara está vacía: no hay vóxeles para calcular los percentiles.")
    lo, hi = np.percentile(v, [low_pct, high_pct])
    return float(lo), float(hi)


def normalize_to_uint8(
    vol: np.ndarray,
    low_pct: float = 0.5,
    high_pct: float = 99.5,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Recorta a [lo, hi] y escala linealmente a 0-255 sobre TODO el volumen a la vez.

    Devuelve `(uint8, {"lo", "hi", "low_pct", "high_pct"})`. Con `(0, 100)` equivale
    a min-max sin recorte. Pase la máscara para excluir el fondo de los percentiles.
    """
    v = np.asarray(vol, dtype=np.float64)
    lo, hi = percentile_bounds(v, low_pct, high_pct, mask)
    if hi <= lo:
        raise ValueError(f"Rango de normalización degenerado: lo={lo}, hi={hi}.")
    scaled = (np.clip(v, lo, hi) - lo) / (hi - lo) * 255.0
    out = np.rint(scaled).astype(np.uint8)
    return out, {"lo": lo, "hi": hi, "low_pct": float(low_pct), "high_pct": float(high_pct)}


def saturation_fraction(vol: np.ndarray, hi: float, mask: np.ndarray | None = None) -> float:
    """Fracción de vóxeles con valor `>= hi` (dentro de la máscara, si se da)."""
    v = np.asarray(vol)
    if mask is not None:
        v = v[np.asarray(mask, dtype=bool)]
    return float(np.mean(v >= hi))
