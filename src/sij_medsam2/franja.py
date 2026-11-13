"""Eje de interlínea y franja articular derivados de los huesos (spec §5.8 punto 8).

Todo es por corte, en 2D, sobre la imagen de 400 px. Fila = índice y; el lado lo fija el ilíaco
(`ilio` = 1 o 3) emparejado con el sacro (2).
"""

from __future__ import annotations

import numpy as np

SEPARACION_MAX_PX = 15
RADIO_FRANJA_PX = 17
HUECO_MAX_FILAS = 3
SACRO = 2


def _por_fila(corte_etiquetas: np.ndarray, ilio: int):
    """Para cada fila: (x_I, x_S, separación) o None si falta algún hueso. Solape: separación 0, eje en
    el punto medio del solape (x_I = x_S = ese punto medio)."""
    H = corte_etiquetas.shape[0]
    out = [None] * H
    for r in range(H):
        ci = np.flatnonzero(corte_etiquetas[r] == ilio)
        cs = np.flatnonzero(corte_etiquetas[r] == SACRO)
        if ci.size == 0 or cs.size == 0:
            continue
        comun = np.intersect1d(ci, cs)
        if comun.size:
            x = (comun[0] + comun[-1]) / 2.0
            out[r] = (x, x, 0.0)
            continue
        d = np.abs(ci[:, None] - cs[None, :])
        i, j = np.unravel_index(np.argmin(d), d.shape)
        out[r] = (float(ci[i]), float(cs[j]), float(d[i, j]))
    return out


def _eje_y_filas(corte_etiquetas: np.ndarray, ilio: int):
    filas = _por_fila(corte_etiquetas, ilio)
    H = len(filas)
    eje = np.full(H, np.nan)
    validas = np.zeros(H, dtype=bool)
    for r, f in enumerate(filas):
        if f is not None and f[2] <= SEPARACION_MAX_PX:
            eje[r] = (f[0] + f[1]) / 2.0
            validas[r] = True
    # huecos de <= HUECO_MAX_FILAS filas entre dos filas válidas: interpolación lineal
    idx = np.flatnonzero(validas)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= HUECO_MAX_FILAS + 1:
            eje[a + 1 : b] = np.interp(np.arange(a + 1, b), [a, b], [eje[a], eje[b]])
    return eje, validas, filas


def eje_interlinea(corte_etiquetas: np.ndarray, ilio: int) -> np.ndarray:
    """Eje L(r) = (x_I + x_S)/2 como float (H,), con NaN en las filas no válidas.

    Válida = ambos huesos presentes y separación <= `SEPARACION_MAX_PX`. Huecos de hasta
    `HUECO_MAX_FILAS` filas entre filas válidas se interpolan; los más largos quedan en NaN. Las filas
    interpoladas cuentan como válidas en `metricas.dist_ejes` y en `frac_filas_discordantes`.
    """
    return _eje_y_filas(corte_etiquetas, ilio)[0]


def franja(corte_etiquetas: np.ndarray, ilio: int) -> np.ndarray:
    """Franja bool (H, W): píxeles a distancia euclídea <= `RADIO_FRANJA_PX` de algún punto del eje,
    dentro de (sacro ∪ ilíaco del lado ∪ hueco entre x_I y x_S) y entre la primera y la última fila válida."""
    H, W = corte_etiquetas.shape
    eje, validas, filas = _eje_y_filas(corte_etiquetas, ilio)
    out = np.zeros((H, W), dtype=bool)
    ok = np.flatnonzero(~np.isnan(eje))
    if ok.size == 0:
        return out
    r0, r1 = np.flatnonzero(validas)[[0, -1]]
    region = (corte_etiquetas == ilio) | (corte_etiquetas == SACRO)
    for r in np.flatnonzero(validas):
        xi, xs, _ = filas[r]
        region[r, int(min(xi, xs)) : int(max(xi, xs)) + 1] = True
    # distancia al eje: comparación directa contra los puntos del eje (exacta)
    pr = ok.astype(float)
    pc = np.clip(np.round(eje[ok]), 0, W - 1)
    rr, cc = np.mgrid[0:H, 0:W]
    cerca = np.zeros((H, W), dtype=bool)
    R2 = RADIO_FRANJA_PX**2
    for a, b in zip(pr, pc):
        cerca |= (rr - a) ** 2 + (cc - b) ** 2 <= R2
    out = cerca & region
    out[:r0] = False
    out[r1 + 1 :] = False
    return out
