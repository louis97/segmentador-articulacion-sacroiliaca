"""Pistas para MedSAM2: cortes de pista, cajas y máscaras, y cortes de evaluación.

Funciones puras. Coordenadas de 400 px; la caja es (x0, y0, x1, y1) con x = columna e y = fila,
el formato que consume `modelo.segmentar_volumen`. Reglas de redondeo y colisiones: spec §5.4 y §5.8.4.
"""

from __future__ import annotations

import numpy as np

TIPOS_PISTA = ("caja", "k1", "k2", "k3")
MARGEN_CAJA_PX = 5
MIN_CORTES_EVAL = 2
HUESOS = (1, 2, 3)  # 1 = ilíaco, 2 = sacro, 3 = ilíaco contralateral


def cortes_pista(nucleo: list[int], tipo: str) -> list[int]:
    """Cortes de pista (ordenados) para un núcleo `nucleo` y un tipo de `TIPOS_PISTA`.

    `caja`: el corte medio `nucleo[len(nucleo)//2]`. `kK`: offsets ⌊(i+1)(|N|-1)/(K+1) + 0.5⌋ (redondeo
    hacia arriba en .5, aritmética entera exacta). Si un offset ya está tomado pasa al siguiente libre
    hacia el final y, si no hay, hacia el inicio.
    """
    if tipo not in TIPOS_PISTA:
        raise ValueError(f"Tipo de pista desconocido: {tipo!r}.")
    nucleo = sorted(int(c) for c in nucleo)
    if not nucleo:
        return []
    if tipo == "caja":
        return [nucleo[len(nucleo) // 2]]
    k = int(tipo[1:])
    m = len(nucleo)
    tomados: list[int] = []
    for i in range(k):
        # floor(x + 0.5) con x = (i+1)(m-1)/(k+1), en enteros: floor((2a + b) / (2b))
        a, b = (i + 1) * (m - 1), k + 1
        off = (2 * a + b) // (2 * b)
        if off in tomados:
            libres_adelante = [o for o in range(off + 1, m) if o not in tomados]
            libres_atras = [o for o in range(off - 1, -1, -1) if o not in tomados]
            candidatos = libres_adelante + libres_atras
            if not candidatos:
                continue  # k > |N|: no caben más pistas
            off = candidatos[0]
        tomados.append(off)
    return sorted(nucleo[o] for o in tomados)


def _caja(m: np.ndarray, margen: int) -> tuple[int, int, int, int]:
    filas = np.flatnonzero(m.any(axis=1))
    cols = np.flatnonzero(m.any(axis=0))
    h, w = m.shape
    return (
        int(max(cols[0] - margen, 0)),
        int(max(filas[0] - margen, 0)),
        int(min(cols[-1] + margen, w - 1)),
        int(min(filas[-1] + margen, h - 1)),
    )


def construir_pistas(mascara: np.ndarray, cortes: list[int], tipo: str) -> dict:
    """Pistas `{corte: {obj_id: {"caja": ...} | {"mascara": ...}}}` desde la máscara experta (D,400,400).

    `tipo == "caja"` da la caja del hueso con margen `MARGEN_CAJA_PX` (recortada a la imagen); `kK` da la
    máscara booleana. Los huesos vacíos en un corte se omiten, y los cortes sin ningún hueso no aparecen.
    """
    if tipo not in TIPOS_PISTA:
        raise ValueError(f"Tipo de pista desconocido: {tipo!r}.")
    pistas: dict = {}
    for c in cortes:
        por_obj = {}
        for h in HUESOS:
            m = mascara[c] == h
            if not m.any():
                continue
            por_obj[h] = {"caja": _caja(m, MARGEN_CAJA_PX)} if tipo == "caja" else {"mascara": m}
        if por_obj:
            pistas[int(c)] = por_obj
    return pistas


def cortes_evaluacion(nucleo: list[int], cortes: list[int]) -> list[int]:
    """Núcleo menos los cortes de pista (ordenado). Ninguna pista puede entrar en las métricas."""
    usados = {int(c) for c in cortes}
    return [int(c) for c in sorted(nucleo) if int(c) not in usados]
