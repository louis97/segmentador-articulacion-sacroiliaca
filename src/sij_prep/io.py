"""Lectura, guardado y utilidades de E/S para los volúmenes de RM."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk

VOXEL_SPACING_XYZ_MM: tuple[float, float, float] = (0.6, 0.6, 3.0)

_RAW_PATTERN = re.compile(
    r"^(?P<prefix>[A-Za-z]+)_(?P<patient>\d+)_(?P<width>\d+)_(?P<height>\d+)"
    r"_(?P<slices>\d+)_(?P<bpp>\d+)_?\.raw$"
)


@dataclass(frozen=True)
class VolumeInfo:
    """Metadatos codificados en el nombre de un archivo .raw."""

    prefix: str
    patient: int
    width: int
    height: int
    slices: int
    bytes_per_pixel: int


def parse_raw_filename(path: Path | str) -> VolumeInfo:
    """Extrae los metadatos del nombre `prefijo_paciente_ancho_alto_cortes_bpp_.raw`."""
    name = Path(path).name
    m = _RAW_PATTERN.match(name)
    if m is None:
        raise ValueError(
            f"El nombre '{name}' no sigue el patrón "
            "'prefijo_paciente_ancho_alto_cortes_bytesPorPixel_.raw'."
        )
    return VolumeInfo(
        prefix=m["prefix"],
        patient=int(m["patient"]),
        width=int(m["width"]),
        height=int(m["height"]),
        slices=int(m["slices"]),
        bytes_per_pixel=int(m["bpp"]),
    )


def list_local_volumes(raw_dir: Path) -> list[Path]:
    """Lista los archivos SIJ_*.raw válidos de un directorio, ordenados por paciente."""
    found: list[tuple[int, Path]] = []
    for p in Path(raw_dir).glob("SIJ_*.raw"):
        try:
            info = parse_raw_filename(p)
        except ValueError:
            continue
        found.append((info.patient, p))
    return [p for _, p in sorted(found, key=lambda t: t[0])]


def read_raw(path: Path) -> np.ndarray:
    """Lee un .raw big-endian uint16 y devuelve un arreglo (D, H, W) en uint16 nativo."""
    path = Path(path)
    info = parse_raw_filename(path)
    expected = info.width * info.height * info.slices * info.bytes_per_pixel
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(
            f"Tamaño inesperado en '{path.name}': {actual} bytes; "
            f"se esperaban {expected} bytes (W*H*D*bpp)."
        )
    data = np.fromfile(path, dtype=">u2")
    return data.reshape(info.slices, info.height, info.width).astype(np.uint16)


def axis_order_check(vol: np.ndarray) -> dict[str, float]:
    """Media de |diferencia| entre vecinos a lo largo de cada eje (cortes, filas, columnas)."""
    v = vol.astype(np.float64)
    return {
        "diff_slices": float(np.abs(np.diff(v, axis=0)).mean()),
        "diff_rows": float(np.abs(np.diff(v, axis=1)).mean()),
        "diff_cols": float(np.abs(np.diff(v, axis=2)).mean()),
    }


def save_volume(
    vol_u8: np.ndarray,
    out_dir: Path,
    stem: str,
    spacing_xyz=VOXEL_SPACING_XYZ_MM,
) -> tuple[Path, Path]:
    """Guarda el volumen uint8 (D, H, W) como .npz y .nii.gz; sobrescribe si existen."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{stem}.npz"
    nii_path = out_dir / f"{stem}.nii.gz"
    np.savez_compressed(npz_path, imgs=vol_u8, spacing=np.array(spacing_xyz))
    img = sitk.GetImageFromArray(vol_u8)
    img.SetSpacing(tuple(float(s) for s in spacing_xyz))
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    sitk.WriteImage(img, str(nii_path))
    return npz_path, nii_path


def load_npz(path) -> tuple[np.ndarray, np.ndarray]:
    """Carga un .npz y devuelve (imgs, spacing)."""
    with np.load(path) as f:
        return f["imgs"], f["spacing"]


def load_nifti(path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Carga un NIfTI y devuelve el arreglo (D, H, W) y el espaciado (x, y, z)."""
    img = sitk.ReadImage(str(path))
    sx, sy, sz = img.GetSpacing()
    return sitk.GetArrayFromImage(img), (float(sx), float(sy), float(sz))


def save_params_json(path: Path, params: dict) -> None:
    """Guarda un diccionario de parámetros como JSON (UTF-8, indent=2)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")


def repo_root(start: Path | None = None) -> Path:
    """Devuelve el primer ancestro (incluido `start`) que contiene pyproject.toml."""
    start = Path.cwd() if start is None else Path(start)
    start = start.resolve()
    for d in (start, *start.parents):
        if (d / "pyproject.toml").is_file():
            return d
    raise FileNotFoundError(
        f"No se encontró pyproject.toml en '{start}' ni en sus directorios padre."
    )
