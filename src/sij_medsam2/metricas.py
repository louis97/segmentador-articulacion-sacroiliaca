"""Métricas por corte: Dice, HD95 y distancia media de borde 2D, y distancia entre ejes de interlínea.

Funciones puras. Bordes 2D = máscara menos su erosión con conectividad 8 (fuera de la imagen cuenta
como fondo). Las distancias son euclídeas exactas entre píxeles de borde.
"""

from __future__ import annotations

import numpy as np

from .franja import eje_interlinea, franja

PX_MM = 0.6
_ESTRUCTURAS = ((1, "iliaco"), (2, "sacro"), (3, "iliaco_contra"))
_LADOS = (("franja_A", 1), ("franja_B", 3))


def dice(a, b) -> float:
    """Dice entre dos máscaras booleanas. NaN si ambas están vacías; 0 si solo una lo está."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    s = a.sum() + b.sum()
    if s == 0:
        return float("nan")
    return float(2.0 * (a & b).sum() / s)


def _borde(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, bool)
    p = np.pad(m, 1, constant_values=False)
    erosion = np.ones_like(m)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            erosion &= p[1 + dy : 1 + dy + m.shape[0], 1 + dx : 1 + dx + m.shape[1]]
    return m & ~erosion


def _distancias_borde(a, b):
    """Distancias de cada punto de borde de a al borde de b, y viceversa (None si alguna está vacía)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    if not a.any() or not b.any():
        return None
    pa, pb = np.argwhere(_borde(a)).astype(float), np.argwhere(_borde(b)).astype(float)
    d = np.sqrt(((pa[:, None, :] - pb[None, :, :]) ** 2).sum(-1))
    return d.min(1), d.min(0)


def hd95_px(a, b) -> float:
    """Percentil 95 de las distancias simétricas borde-borde, en px. NaN si alguna máscara está vacía."""
    d = _distancias_borde(a, b)
    if d is None:
        return float("nan")
    return float(np.percentile(np.concatenate(d), 95))


def dist_media_borde_px(a, b) -> float:
    """Media de las distancias simétricas borde-borde, en px. NaN si alguna máscara está vacía."""
    d = _distancias_borde(a, b)
    if d is None:
        return float("nan")
    return float(np.concatenate(d).mean())


def dist_ejes(eje_pred: np.ndarray, eje_exp: np.ndarray) -> dict:
    """Distancia |L_pred - L_exp| en las filas válidas en ambos: `mediana_px`, `p95_px` y `frac_filas_discordantes`.

    `frac_filas_discordantes` = (filas válidas solo en uno de los dos) / (filas válidas en alguno, la unión);
    0 = coincidencia total, 1 = ninguna fila en común, NaN si ninguno tiene filas válidas (spec §5.8.8).
    Mediana y p95 son NaN sin filas comunes. Las filas interpoladas por `eje_interlinea` (huecos de
    <= 3 filas) cuentan como válidas, tanto aquí como en la fracción.
    """
    vp, ve = ~np.isnan(eje_pred), ~np.isnan(eje_exp)
    ambos, union = vp & ve, vp | ve
    frac_filas_discordantes = float((vp ^ ve).sum() / union.sum()) if union.any() else float("nan")
    if not ambos.any():
        return {"mediana_px": float("nan"), "p95_px": float("nan"), "frac_filas_discordantes": frac_filas_discordantes}
    d = np.abs(eje_pred[ambos] - eje_exp[ambos])
    return {"mediana_px": float(np.median(d)), "p95_px": float(np.percentile(d, 95)), "frac_filas_discordantes": frac_filas_discordantes}


def _fila(estructura, **kw) -> dict:
    nan = float("nan")
    base = dict(estructura=estructura, dice=nan, hd95_px=nan, hd95_mm=nan, dist_media_px=nan,
                eje_mediana_px=nan, eje_p95_px=nan, frac_filas_discordantes=nan, vacio_pred=False, vacio_exp=False)
    base.update(kw)
    return base


def metricas_corte(pred: np.ndarray, exp: np.ndarray) -> list[dict]:
    """Filas planas de un corte: una por hueso (iliaco, sacro, iliaco_contra) y una por franja
    (franja_A = sacro con ilíaco 1, franja_B = sacro con ilíaco 3). `pred` y `exp`: etiquetas (H, W).

    Claves: estructura, dice, hd95_px, hd95_mm, dist_media_px, eje_mediana_px, eje_p95_px, frac_filas_discordantes,
    vacio_pred, vacio_exp. En los huesos los campos de eje son NaN; en las franjas hd95 y dist_media son
    NaN y dice es el de la franja. `vacio_*` marca máscara (o franja) vacía en predicción / experto.
    """
    filas = []
    for etiqueta, nombre in _ESTRUCTURAS:
        p, e = pred == etiqueta, exp == etiqueta
        h = hd95_px(p, e)
        filas.append(_fila(nombre, dice=dice(p, e), hd95_px=h, hd95_mm=h * PX_MM,
                           dist_media_px=dist_media_borde_px(p, e),
                           vacio_pred=not p.any(), vacio_exp=not e.any()))
    for nombre, ilio in _LADOS:
        fp, fe = franja(pred, ilio), franja(exp, ilio)
        de = dist_ejes(eje_interlinea(pred, ilio), eje_interlinea(exp, ilio))
        filas.append(_fila(nombre, dice=dice(fp, fe), eje_mediana_px=de["mediana_px"],
                           eje_p95_px=de["p95_px"], frac_filas_discordantes=de["frac_filas_discordantes"],
                           vacio_pred=not fp.any(), vacio_exp=not fe.any()))
    return filas
