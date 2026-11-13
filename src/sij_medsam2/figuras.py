"""Figuras de resumen de MedSAM2 (en español), a partir de `resultados/medsam2/<dataset>/`."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from sij_prep.io import repo_root
from sij_prep.viz import _CAT, _INK, _contour, _img_axes, _style_axes, _subplots

from .datos import stir_preprocesada
from .pipeline import ESTRUCTURAS, METRICAS, _por_caso, _por_paciente, _ruta_pred
from .registro import mascara_experta

_NOMBRES = {"iliaco": "Ilíaco", "sacro": "Sacro", "iliaco_contra": "Ilíaco contralateral",
            "franja_A": "Franja A (sacro-ilíaco 1)", "franja_B": "Franja B (sacro-ilíaco 3)"}
_TIPOS = {"caja": "Caja", "k1": "1 máscara", "k2": "2 máscaras", "k3": "3 máscaras"}


def _leer_cortes(ruta: Path) -> list[dict]:
    filas = []
    with open(ruta, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k in ("n", "corte", "distancia_a_pista"):
                r[k] = int(r[k])
            for m in METRICAS:
                r[m] = float(r[m])
            for k in ("vacio_pred", "vacio_exp"):
                r[k] = bool(int(r[k]))
            filas.append(r)
    return filas


def _valores_paciente(filas: list[dict], tipo: str, est: str, metrica: str) -> list[float]:
    pac = _por_paciente(_por_caso([f for f in filas if f["tipo"] == tipo]))
    return [p[metrica] for p in pac if p["estructura"] == est and not math.isnan(p[metrica])]


def _pacientes_comunes(filas: list[dict], tipos) -> set:
    """Pacientes con métricas en todos los tipos de pista (el conjunto `comun` de `resumen.csv`)."""
    por_tipo = [{f["paciente"] for f in filas if f["tipo"] == t} for t in tipos]
    return set.intersection(*por_tipo) if por_tipo else set()


def _error_por_tipo(filas, tipos, ruta: Path) -> Path:
    """Dice y error del eje por tipo de pista, solo con el subconjunto común de pacientes (spec §5.8.10)."""
    comunes = _pacientes_comunes(filas, tipos)
    filas = [f for f in filas if f["paciente"] in comunes]
    fig, (a, b) = _subplots(1, 2, figsize=(13, 5))
    rng = np.random.default_rng(0)
    ancho = 0.8 / len(ESTRUCTURAS)
    for i, est in enumerate(ESTRUCTURAS):
        x0 = np.arange(len(tipos)) - 0.4 + ancho * (i + 0.5)
        vals = [_valores_paciente(filas, t, est, "dice") for t in tipos]
        a.errorbar(x0, [np.mean(v) if v else np.nan for v in vals], [np.std(v, ddof=1) if len(v) > 1 else 0 for v in vals],
                   fmt="o", color=_CAT[i], capsize=3, label=_NOMBRES[est], zorder=3)
        for x, v in zip(x0, vals):
            a.scatter(x + rng.uniform(-ancho / 3, ancho / 3, len(v)), v, s=8, color=_CAT[i], alpha=0.35, zorder=2)
    a.set_xticks(range(len(tipos)), [_TIPOS.get(t, t) for t in tipos])
    a.set_ylim(0, 1.02)
    a.set_xlabel("Tipo de pista")
    a.set_ylabel("Dice (media ± desv. entre pacientes)")
    a.set_title(f"Dice por estructura y tipo de pista\n(subconjunto común: {len(comunes)} pacientes; puntos = pacientes)")
    a.legend(fontsize=8, loc="lower right")
    _style_axes(a)

    for i, est in enumerate(("franja_A", "franja_B")):
        x0 = np.arange(len(tipos)) - 0.2 + 0.4 * i
        for métrica, marc, nombre in (("eje_mediana_px", "o", "mediana"), ("eje_p95_px", "^", "p95")):
            vals = [_valores_paciente(filas, t, est, métrica) for t in tipos]
            b.errorbar(x0 + (0.06 if marc == "^" else -0.06), [np.mean(v) if v else np.nan for v in vals],
                       [np.std(v, ddof=1) if len(v) > 1 else 0 for v in vals], fmt=marc,
                       color=_CAT[3 + i], alpha=1.0 if marc == "o" else 0.55, capsize=3,
                       label=f"{_NOMBRES[est]}, {nombre}")
    b.set_xticks(range(len(tipos)), [_TIPOS.get(t, t) for t in tipos])
    b.set_xlabel("Tipo de pista")
    b.set_ylabel("Distancia entre ejes de interlínea (píxeles; 1 px = 0,6 mm)")
    b.set_title(f"Distancia entre ejes por tipo de pista\n(subconjunto común: {len(comunes)} pacientes; media ± desv.)")
    b.legend(fontsize=8)
    _style_axes(b)
    fig.suptitle("MedSAM2 frente a la máscara experta, cortes de evaluación (sin cortes de pista)", fontsize=12)
    fig.savefig(ruta, dpi=130)
    return ruta


MIN_PACIENTES_PUNTO = 3


def _error_vs_distancia(filas, tipos, ruta: Path) -> Path:
    """Error frente a la distancia a la pista, solo con el subconjunto común de pacientes."""
    comunes = _pacientes_comunes(filas, tipos)
    filas = [f for f in filas if f["paciente"] in comunes]
    fig, (a, b) = _subplots(1, 2, figsize=(13, 5.5))
    omitidos = 0
    for i, t in enumerate(tipos):
        ft = [f for f in filas if f["tipo"] == t]
        for ax, ests, metrica in ((a, ("iliaco", "sacro", "iliaco_contra"), "dice"),
                                  (b, ("franja_A", "franja_B"), "eje_mediana_px")):
            xs, ys, es, ns = [], [], [], []
            for d in sorted({f["distancia_a_pista"] for f in ft}):
                sel = [f for f in ft if f["distancia_a_pista"] == d and f["estructura"] in ests]
                pac = _por_paciente(_por_caso(sel))
                por_pac: dict = {}
                for p in pac:
                    if not math.isnan(p[metrica]):
                        por_pac.setdefault(p["paciente"], []).append(p[metrica])
                v = [np.mean(x) for x in por_pac.values()]  # un valor por paciente (promedia estructuras)
                if len(v) < MIN_PACIENTES_PUNTO:
                    omitidos += len(v) > 0
                    continue
                xs.append(d); ys.append(np.mean(v)); es.append(np.std(v, ddof=1)); ns.append(len(v))
            off = (i - (len(tipos) - 1) / 2) * 0.06
            ax.errorbar(np.array(xs) + off, ys, es, fmt="o-", color=_CAT[i], capsize=3, label=_TIPOS.get(t, t))
            for x, y, n in zip(xs, ys, ns):
                ax.annotate(f"n={n}", (x + off, y), textcoords="offset points", xytext=(5, 5), fontsize=7, color=_CAT[i])
    a.set_ylabel("Dice (media de los 3 huesos y de los pacientes)")
    a.set_title("Dice frente a la distancia al corte de pista")
    b.set_ylabel("Distancia entre ejes, mediana (px)")
    b.set_title("Distancia entre ejes frente a la distancia al corte de pista")
    for ax in (a, b):
        ax.set_xlabel("Distancia al corte de pista más cercano (cortes; 1 corte = 3 mm)")
        ax.set_xticks(sorted({f["distancia_a_pista"] for f in filas}))
        ax.legend(title="Tipo de pista", fontsize=8)
        _style_axes(ax)
    fig.suptitle(f"Subconjunto común ({len(comunes)} pacientes). Barras = desv. estándar entre pacientes; n = pacientes por punto; "
                 f"se omiten los puntos con menos de {MIN_PACIENTES_PUNTO} pacientes", fontsize=10)
    fig.savefig(ruta, dpi=130)
    return ruta


def _elegir_ejemplos(por: dict) -> list[tuple[int, int, str]]:
    """Mediano, mejor y peor corte por Dice de sacro (`por[(n, corte)]`), con casos distintos.

    Se ordenan todos los cortes por (Dice, n, corte) ascendente (desempate por n y corte). Mediano =
    elemento central (índice len//2), mejor = último, peor = primero. Si un caso ya fue elegido, se
    toma el siguiente corte de otro caso en el orden, alejándose del objetivo (hacia el centro para
    el mejor y el peor: el más cercano en rango).
    """
    orden = sorted(por, key=lambda k: (por[k], k[0], k[1]))
    objetivos = (("mediano", len(orden) // 2), ("mejor", len(orden) - 1), ("peor", 0))
    elegidos, vistos = [], set()
    for rol, idx in objetivos:
        for j in sorted(range(len(orden)), key=lambda j: (abs(j - idx), j)):
            n, c = orden[j]
            if n not in vistos:
                elegidos.append((n, c, rol))
                vistos.add(n)
                break
    return elegidos


def _ejemplos(filas, dataset, ruta: Path, tipo: str = "k1") -> Path:
    """3 casos distintos: corte de Dice de sacro mediano, mejor y peor del tipo `tipo`."""
    por = {(f["n"], f["corte"]): f["dice"] for f in filas
           if f["tipo"] == tipo and f["estructura"] == "sacro" and not math.isnan(f["dice"])}
    if not por:
        raise ValueError(f"No hay métricas del tipo '{tipo}' para los ejemplos.")
    elegidos = _elegir_ejemplos(por)
    fig, axes = _subplots(1, len(elegidos), figsize=(5.5 * len(elegidos), 6.3), squeeze=False)
    color = {1: _CAT[0], 2: _CAT[1], 3: _CAT[2]}
    for ax, (n, c, rol) in zip(axes[0], elegidos):
        img = stir_preprocesada(n, dataset)[c]
        exp = mascara_experta(n)[c]
        with np.load(_ruta_pred(dataset, n, tipo)) as f:
            pred = f["pred"][c]
        ax.imshow(img, cmap="gray", origin="upper")
        for h in (1, 2, 3):
            rel = np.ma.masked_where(pred != h, np.ones_like(pred, float))
            ax.imshow(rel, cmap=ListedColormap([color[h]]), alpha=0.4, origin="upper")
            _contour(ax, exp == h, color[h], lw=1.6)
        ax.set_title(f"Dice de sacro {rol}: {por[(n, c)]:.2f}\nCaso n = {n}, corte {c}", fontsize=10, color=_INK)
        _img_axes(ax)
    leyenda = [Patch(facecolor=color[1], alpha=0.4, label="Predicción: ilíaco"),
               Patch(facecolor=color[2], alpha=0.4, label="Predicción: sacro"),
               Patch(facecolor=color[3], alpha=0.4, label="Predicción: ilíaco contralateral"),
               Line2D([], [], color=_INK, lw=1.6, label="Contorno (color del hueso) = máscara experta")]
    axes[0][-1].legend(handles=leyenda, loc="lower right", fontsize=7, framealpha=0.9)
    fig.suptitle(f"Pista «{_TIPOS.get(tipo, tipo)}»: cortes de evaluación con Dice de sacro mediano, mejor y peor "
                 "(orden por Dice, n y corte; si un caso se repite se toma el siguiente de otro caso más cercano en rango)",
                 fontsize=9)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=130)
    return ruta


def resumen(dir_resultados: Path) -> list[Path]:
    """Genera `error_por_tipo.png` y `error_vs_distancia.png` en `dir_resultados` y `ejemplos.png` en
    `data/processed/medsam2/<dataset>/figuras/` (ignorado por git). Devuelve las 3 rutas.

    No necesita el modelo ni el dispositivo: solo lee los CSV, la configuración y las predicciones
    cacheadas. Para regenerar solo las figuras se llama a esta función, no a `ejecutar_pipeline`.
    """
    d = Path(dir_resultados)
    filas = _leer_cortes(d / "metricas_por_corte.csv")
    cfg = json.loads((d / "configuracion.json").read_text(encoding="utf-8"))
    tipos = [t for t in cfg["parametros"]["tipos"] if any(f["tipo"] == t for f in filas)]
    return [_error_por_tipo(filas, tipos, d / "error_por_tipo.png"),
            _error_vs_distancia(filas, tipos, d / "error_vs_distancia.png"),
            _ejemplos(filas, cfg["dataset"], repo_root() / f"data/processed/medsam2/{cfg['dataset']}/figuras/ejemplos.png",
                       "k1" if "k1" in tipos else tipos[0])]
