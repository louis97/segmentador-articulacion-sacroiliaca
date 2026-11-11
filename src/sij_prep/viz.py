"""Gráficas de control para cada paso de la preparación de imágenes.

Todas las funciones devuelven un `matplotlib.figure.Figure` (API orientada a
objetos, sin mostrar la figura), con textos en español e imágenes en escala de grises
con `origin="upper"`.
"""
from __future__ import annotations

import math

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from sij_prep.intensity import percentile_bounds, saturation_fraction
from sij_prep.io import VOXEL_SPACING_XYZ_MM
from sij_prep.model_view import denormalize_for_display

# Colores categóricos (Okabe-Ito, seguros para daltonismo), en orden fijo.
_CAT = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
_BEFORE = "#0072B2"
_AFTER = "#D55E00"
_MARK = "#E69F00"  # marcadores sobre imágenes en gris (líneas, cuadrículas)
_INK = "#222222"
_MUTED = "#666666"
_GRID = "#DDDDDD"

_IMG = dict(cmap="gray", origin="upper")


def _new_figure(figsize) -> Figure:
    """`Figure` con layout constrained y lienzo Agg propio (independiente de pyplot)."""
    fig = Figure(figsize=figsize, layout="constrained")
    FigureCanvasAgg(fig)
    return fig


def _subplots(*args, figsize, **kw):
    """Crea `Figure` y ejes sin pasar por pyplot (no deja estado global)."""
    fig = _new_figure(figsize)
    return fig, fig.subplots(*args, **kw)


def _style_axes(ax) -> None:
    """Estilo común para ejes de datos: sin marco superior/derecho y rejilla tenue."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _img_axes(ax, xlabel: str = "columna (píxel)", ylabel: str = "fila (píxel)") -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _contour(ax, mask2d: np.ndarray, color: str, lw: float = 1.5, ls: str = "-") -> None:
    if mask2d.any():
        ax.contour(mask2d.astype(float), levels=[0.5], colors=[color], linewidths=lw, linestyles=ls)


def plot_mosaic(vol: np.ndarray, title: str, ncols: int = 6) -> Figure:
    """Mosaico con todos los cortes del volumen, cada uno con su índice."""
    d = vol.shape[0]
    nrows = math.ceil(d / ncols)
    fig, axes = _subplots(
        nrows, ncols, figsize=(2.3 * ncols, 2.4 * nrows + 0.6), squeeze=False
    )
    vmin, vmax = float(np.min(vol)), float(np.percentile(vol, 99.9))
    for k, ax in enumerate(axes.ravel()):
        if k < d:
            ax.imshow(vol[k], vmin=vmin, vmax=vmax, **_IMG)
            ax.set_title(f"corte {k}", fontsize=9, color=_INK)
        ax.axis("off")
    fig.suptitle(f"{title}: {d} cortes (misma escala de intensidad)", fontsize=12)
    return fig


def plot_orthogonal_check(
    vol: np.ndarray, spacing_xyz=VOXEL_SPACING_XYZ_MM, row: int | None = None
) -> Figure:
    """Corte central con la fila `row` marcada y la reconstrucción `vol[:, row, :]` estirada."""
    d, h, w = vol.shape
    row = h // 2 if row is None else row
    mid = d // 2
    aspect = spacing_xyz[2] / spacing_xyz[0]
    fig, (a, b) = _subplots(
        1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1, 1]}
    )
    a.imshow(vol[mid], **_IMG)
    a.axhline(row, color=_MARK, linewidth=1.5, label=f"fila {row} (reconstrucción)")
    a.set_title(f"Corte coronal oblicuo central (corte {mid})")
    _img_axes(a)
    a.legend(loc="lower right", fontsize=8, framealpha=0.9)

    b.imshow(vol[:, row, :], aspect=aspect, **_IMG)
    b.set_title(
        f"Reconstrucción en el plano perpendicular\nvol[:, {row}, :] ({d}×{w})\n"
        f"estirada ×{aspect:g} ({spacing_xyz[2]:g} mm / {spacing_xyz[0]:g} mm):\n"
        "cada corte se ve como un escalón",
        fontsize=10,
    )
    b.set_xlabel("columna (píxel)")
    b.set_ylabel(f"índice de corte ({spacing_xyz[2]:g} mm por corte)")
    b.set_yticks(range(0, d, max(1, d // 6)))
    return fig


def plot_n4_trio(
    original: np.ndarray,
    bias_field: np.ndarray,
    corrected: np.ndarray,
    mask: np.ndarray,
    slice_idx: int,
    line_row: int | None = None,
) -> Figure:
    """Original con contorno de la máscara, campo de sesgo, corregido y perfil de intensidad."""
    h = original.shape[1]
    line_row = h // 2 if line_row is None else line_row
    o, bf, c = original[slice_idx], bias_field[slice_idx], corrected[slice_idx]
    m = mask[slice_idx].astype(bool)
    # misma escala para original y corregido, para que la comparación sea honesta
    vmax = float(np.percentile(np.concatenate([o[m], c[m]]) if m.any() else o, 99.5))

    fig, axes = _subplots(2, 2, figsize=(11, 10))
    (a, b), (cax, p) = axes
    a.imshow(o, vmin=0, vmax=vmax, **_IMG)
    _contour(a, m, _MARK)
    a.axhline(line_row, color="#56B4E9", linewidth=1.8, linestyle="--")
    a.set_title(f"Original (corte {slice_idx}); contorno = máscara,\nlínea punteada = fila {line_row}")
    _img_axes(a)

    im = b.imshow(bf, cmap="magma", origin="upper")
    b.set_title("Campo de sesgo estimado (factor multiplicativo)")
    _img_axes(b)
    fig.colorbar(im, ax=b, shrink=0.8, label="factor multiplicativo (1 = sin cambio)")

    cax.imshow(c, vmin=0, vmax=vmax, **_IMG)
    cax.set_title("Corregida con N4 (misma escala que la original)")
    _img_axes(cax)

    p.plot(o[line_row], color=_BEFORE, linewidth=1.2, label="antes (original)")
    p.plot(c[line_row], color=_AFTER, linewidth=1.2, label="después (N4)")
    p.set_title(f"Perfil de intensidad a lo largo de la fila {line_row}")
    p.set_xlabel("columna (píxel)")
    p.set_ylabel("intensidad (unidades del equipo)")
    p.legend(fontsize=9)
    _style_axes(p)
    return fig


def plot_mask_comparison(vol: np.ndarray, masks: dict[str, np.ndarray], slice_idx: int) -> Figure:
    """Contornos de cada máscara superpuestos sobre el mismo corte."""
    fig, (a, b) = _subplots(1, 2, figsize=(11, 5.5))
    s = vol[slice_idx]
    vmax = float(np.percentile(s, 99.5))
    a.imshow(s, vmin=0, vmax=vmax, **_IMG)
    a.set_title(f"Corte {slice_idx} sin contornos")
    _img_axes(a)
    b.imshow(s, vmin=0, vmax=vmax, **_IMG)
    styles = ["-", "--", ":", "-."]
    handles = []
    for i, (name, m) in enumerate(masks.items()):
        col = _CAT[i % len(_CAT)]
        _contour(b, m[slice_idx], col, lw=1.8, ls=styles[i % 4])
        frac = 100 * float(m[slice_idx].mean())
        handles.append(
            Line2D([0], [0], color=col, linestyle=styles[i % 4], lw=2,
                       label=f"{name} ({frac:.1f} % del corte)")
        )
    b.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)
    b.set_title(f"Contorno de cada máscara (corte {slice_idx})")
    _img_axes(b)
    return fig


def plot_clipping_comparison(
    vol: np.ndarray,
    criteria: list[tuple[float, float]],
    slice_idx: int,
    mask: np.ndarray | None = None,
) -> Figure:
    """Un corte por criterio de percentiles `(bajo, alto)`, con los píxeles saturados
    (>= hi) resaltados, el % de saturación y un histograma con la línea `hi` de cada criterio."""
    n = len(criteria)
    fig = _new_figure((4.4 * n, 9.2))
    gs = fig.add_gridspec(2, n, height_ratios=[1.15, 1])
    s = vol[slice_idx]
    v = vol[mask.astype(bool)] if mask is not None else vol.ravel()
    donde = "dentro de la máscara" if mask is not None else "sobre todo el volumen"

    hist_ax = fig.add_subplot(gs[1, :])
    hist_ax.hist(v, bins=200, color="#BBBBBB", log=True)
    for i, (lp, hp) in enumerate(criteria):
        lo, hi = percentile_bounds(vol, lp, hp, mask)
        sat = 100 * saturation_fraction(vol, hi, mask)
        col = _CAT[i % len(_CAT)]
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(s, vmin=lo, vmax=hi, **_IMG)
        sat_px = np.ma.masked_where(s < hi, np.ones(s.shape))
        ax.imshow(sat_px, cmap=ListedColormap(["#FF2D95"]), alpha=0.75, origin="upper", vmin=0, vmax=1)
        ax.set_title(
            f"Percentiles {lp:g}–{hp:g}  (hi = {hi:.0f})\n"
            f"saturación alta: {sat:.2f} %",
            fontsize=10, color=col, fontweight="bold",
        )
        for sp in ax.spines.values():
            sp.set_edgecolor(col)
            sp.set_linewidth(3)
        _img_axes(ax)
        hist_ax.axvline(hi, color=col, linewidth=2, linestyle=["-", "--", ":"][i % 3],
                        label=f"hi, percentil {hp:g} = {hi:.0f}")
    hist_ax.set_title(f"Histograma del volumen ({donde}) y valor de recorte alto de cada criterio")
    hist_ax.set_xlabel("intensidad (unidades del equipo)")
    hist_ax.set_ylabel("número de vóxeles (escala log)")
    handles, _ = hist_ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="#FF2D95", alpha=0.75, label="píxeles saturados (≥ hi), en las imágenes"))
    hist_ax.legend(handles=handles, fontsize=9, loc="upper left")
    _style_axes(hist_ax)
    fig.suptitle(f"Comparación de criterios de recorte (corte {slice_idx}); magenta = píxeles saturados (≥ hi); "
                 f"% de saturación calculado {donde}",
                 fontsize=12)
    return fig


def plot_histograms_before_after(
    vol_before: np.ndarray,
    vol_after_u8: np.ndarray,
    lo: float,
    hi: float,
    mask: np.ndarray | None = None,
) -> Figure:
    """Histogramas antes (con `lo` y `hi`) y después de normalizar a uint8; `mask` excluye el fondo."""
    fig, (a, b) = _subplots(1, 2, figsize=(11, 4.5))
    if mask is not None:
        m = mask.astype(bool)
        before, after = vol_before[m], vol_after_u8[m]
        nota = " (dentro de la máscara)"
    else:
        before, after = vol_before.ravel(), vol_after_u8.ravel()
        nota = " (incluye el fondo)"
    a.hist(before, bins=200, color=_BEFORE, log=True)
    a.axvline(lo, color=_MARK, linewidth=2, linestyle="--", label=f"lo = {lo:.0f}")
    a.axvline(hi, color=_AFTER, linewidth=2, label=f"hi = {hi:.0f}")
    a.set_title(f"Antes de normalizar{nota}")
    a.set_xlabel("intensidad (unidades del equipo)")
    a.set_ylabel("número de vóxeles (escala log)")
    a.legend(fontsize=9)
    _style_axes(a)
    b.hist(after, bins=64, range=(0, 256), color=_AFTER, log=True)
    b.set_title(f"Después de normalizar (uint8){nota}")
    b.set_xlabel("intensidad (0–255)")
    b.set_ylabel("número de vóxeles (escala log)")
    _style_axes(b)
    fig.suptitle("Histogramas del volumen antes y después de la normalización", fontsize=12)
    return fig


def plot_volume_distributions(
    vols: dict[str, np.ndarray],
    title: str,
    masks: dict[str, np.ndarray] | None = None,
    ylabel: str | None = None,
) -> Figure:
    """Diagramas de caja por volumen (sin atípicos) sobre una muestra fija de vóxeles.

    Con `masks` (mismas claves que `vols`) se muestrean solo los vóxeles dentro de la máscara.
    Si `ylabel` es None se deduce del dtype (uint8: 0–255; otro: unidades del equipo).
    """
    rng = np.random.default_rng(0)
    n_sample = 100_000
    data = []
    for k, v in vols.items():
        v = np.asarray(v)
        flat = v[masks[k].astype(bool)] if masks is not None else v.ravel()
        idx = rng.choice(flat.size, size=min(n_sample, flat.size), replace=False)
        data.append(flat[idx])
    if ylabel is None:
        es_u8 = all(np.asarray(v).dtype == np.uint8 for v in vols.values())
        ylabel = "intensidad (0–255)" if es_u8 else "intensidad (unidades del equipo)"
    fondo = "dentro de la máscara" if masks is not None else "incluye el fondo"
    fig, ax = _subplots(figsize=(max(6.5, 1.0 * len(vols) + 2), 4.8))
    bp = ax.boxplot(data, tick_labels=list(vols.keys()), showfliers=False, patch_artist=True,
                    medianprops=dict(color=_INK, linewidth=1.5))
    for box in bp["boxes"]:
        box.set(facecolor=_BEFORE, alpha=0.55, edgecolor=_BEFORE)
    ax.set_title(f"{title}\n(muestra fija de {n_sample:,} vóxeles por volumen; sin valores atípicos; {fondo})".replace(",", "."))
    ax.set_xlabel("volumen")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    _style_axes(ax)
    return fig


def plot_model_input(slice_u8: np.ndarray, model_input: np.ndarray) -> Figure:
    """Corte original, cada canal normalizado (ImageNet) con sus estadísticas y la imagen desnormalizada."""
    fig, axes = _subplots(1, 5, figsize=(19, 5.0))
    a = axes[0]
    a.imshow(slice_u8, vmin=0, vmax=255, **_IMG)
    a.set_title(f"Original uint8\n{slice_u8.shape[1]}×{slice_u8.shape[0]} píxeles")
    _img_axes(a)
    ch, mh, mw = model_input.shape
    vmin, vmax = float(model_input.min()), float(model_input.max())
    nombres = ["R", "G", "B"]
    for c in range(3):
        ax = axes[1 + c]
        ch_v = model_input[c]
        im = ax.imshow(ch_v, vmin=vmin, vmax=vmax, **_IMG)
        ax.set_title(
            f"Canal {nombres[c]} normalizado, {mw}×{mh} píxeles\n"
            f"mín {ch_v.min():.2f} · máx {ch_v.max():.2f} · media {ch_v.mean():.2f}",
            fontsize=10,
        )
        _img_axes(ax)
    fig.colorbar(im, ax=axes[3], shrink=0.8, label="valor normalizado (ImageNet)")
    d = axes[4]
    d.imshow(denormalize_for_display(model_input), origin="upper")
    d.set_title("Desnormalizada (3 canales)\npara visualizar, en [0, 1]")
    _img_axes(d)
    fig.suptitle("Entrada de MedSAM2: los 3 canales contienen la MISMA imagen; solo cambian las constantes "
                 "de normalización (media y desviación de ImageNet) de cada canal", fontsize=12)
    return fig


def plot_key_slice(vol_u8: np.ndarray, idx: int, scores: np.ndarray, grid_step: int = 50,
                   profile_is_fraction: bool = True) -> Figure:
    """Perfil descriptivo por corte, con el corte sugerido y su cuadrícula en píxeles.

    Si ``profile_is_fraction`` es False, el perfil es la intensidad media (0-255) y no una fracción.
    """
    d, h, w = vol_u8.shape
    fig, (a, b) = _subplots(1, 2, figsize=(12, 5.2),
                               gridspec_kw={"width_ratios": [1.1, 1]})
    x = np.arange(len(scores))
    perfil = 100 * np.asarray(scores) if profile_is_fraction else np.asarray(scores)
    a.plot(x, perfil, color=_BEFORE, marker="o", markersize=5, linewidth=1.5,
           label="perfil descriptivo (no es un puntaje de selección)")
    a.axvline(idx, color=_AFTER, linewidth=2, label=f"corte sugerido = {idx} (corte central por convención)")
    a.set_title("Perfil descriptivo por corte")
    a.set_xlabel("índice de corte")
    if profile_is_fraction:
        a.set_ylabel("fracción del corte dentro de la máscara corporal (%)")
        a.set_ylim(0, 100)
    else:
        a.set_ylabel("intensidad media del corte")
    a.set_xticks(x)
    a.legend(fontsize=8, loc="best")
    _style_axes(a)

    b.imshow(vol_u8[idx], vmin=0, vmax=255, **_IMG)
    b.set_xticks(np.arange(0, w + 1, grid_step))
    b.set_yticks(np.arange(0, h + 1, grid_step))
    b.grid(True, color=_MARK, linewidth=0.7, alpha=0.8)
    b.set_title(f"Corte sugerido {idx} con cuadrícula cada {grid_step} píxeles\n"
                "(la elección final la hace una persona)")
    _img_axes(b, "columna (píxel del corte original)", "fila (píxel del corte original)")
    return fig


def plot_central_slices(vols: dict[str, np.ndarray], title: str) -> Figure:
    """Corte central (D // 2) de cada volumen, con escala de intensidad propia por volumen."""
    n = len(vols)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = _subplots(nrows, ncols, figsize=(4.6 * ncols, 4.8 * nrows), squeeze=False)
    flat = axes.ravel()
    for ax, (name, v) in zip(flat, vols.items()):
        d = v.shape[0]
        ax.imshow(v[d // 2], vmin=float(v.min()), vmax=float(v.max()), **_IMG)
        ax.set_title(f"{name} — corte {d // 2} de {d}", fontsize=10)
        _img_axes(ax)
    for ax in flat[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    return fig
