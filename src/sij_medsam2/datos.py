"""Datos de MedSAM2: descarga de Kaggle, selección de casos y STIR preprocesada.

Dos conjuntos de origen: `kaggle` (STIR/ de Kaggle) y `mendeley` (data/raw/).
Reutiliza `sij_prep` para lectura, máscara corporal, N4 y normalización.
"""
from __future__ import annotations

import json
import shutil
import urllib.request
import warnings
import zipfile
from pathlib import Path
from typing import Literal

import numpy as np

from sij_prep.intensity import body_mask, n4_correct, normalize_to_uint8
from sij_prep.io import parse_raw_filename, read_raw, repo_root

KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/krzysztofrzecki/bone-marrow-oedema-data"
N_RAW_KAGGLE = 156
_CARPETA_KAGGLE = "bone-marrow-oedema-data"
_PARAMS_JSON = "data/processed/preprocessing_params.json"

# Parámetros de preprocesado de la iteración 1 (idénticos a data/processed/preprocessing_params.json).
# Son constantes del código para que un clon limpio funcione sin `data/`.
PREPROCESADO = {
    "clip": {"low_pct": 0.5, "high_pct": 99.5},
    "mask_method": "low_threshold",
    "n4": {"shrink_factor": 4, "max_iterations": [50, 50, 50]},
    "spacing_xyz_mm": [0.6, 0.6, 3.0],
}


def _params_preprocesado() -> dict:
    """Devuelve `PREPROCESADO`; si existe el JSON de la iteración 1 y difiere, avisa (no falla si no existe)."""
    ruta = repo_root() / _PARAMS_JSON
    if ruta.is_file():
        en_disco = json.loads(ruta.read_text(encoding="utf-8"))
        dif = [k for k in PREPROCESADO if en_disco.get(k) != PREPROCESADO[k]]
        if dif:
            warnings.warn(f"{_PARAMS_JSON} difiere de las constantes de sij_medsam2.datos.PREPROCESADO en {dif}; "
                          "se usan las constantes.")
    return PREPROCESADO


def _raws_kaggle(carpeta: Path) -> list[Path]:
    return sorted(carpeta.rglob("*.raw"))


def asegurar_kaggle(dest: Path | None = None) -> Path:
    """Garantiza que Kaggle esté descomprimido en `dest` y devuelve `bone-marrow-oedema-data`.

    Si ya están los 156 `.raw` no hace nada; si faltan, descarga el zip a un `.part`,
    lo descomprime y comprueba el conteo.
    """
    dest = Path(dest) if dest is not None else repo_root() / "data/external/kaggle-bmo"
    carpeta = dest / _CARPETA_KAGGLE
    if len(_raws_kaggle(carpeta)) == N_RAW_KAGGLE:
        return carpeta
    dest.mkdir(parents=True, exist_ok=True)
    zip_final = dest / "archive.zip"
    if not zip_final.exists():
        part = dest / "archive.zip.part"
        try:
            with urllib.request.urlopen(KAGGLE_URL, timeout=60) as resp, open(part, "wb") as f:
                shutil.copyfileobj(resp, f, 1 << 20)
            part.replace(zip_final)
        except Exception as exc:
            part.unlink(missing_ok=True)
            raise RuntimeError(
                f"No se pudo descargar Kaggle desde {KAGGLE_URL}: {exc}. "
                f"Descargue el zip manualmente y colóquelo en {zip_final}."
            ) from exc
    with zipfile.ZipFile(zip_final) as z:
        con_raiz = all(n.startswith(_CARPETA_KAGGLE + "/") for n in z.namelist())
        z.extractall(dest if con_raiz else carpeta)
    n = len(_raws_kaggle(carpeta))
    if n != N_RAW_KAGGLE:
        raise RuntimeError(f"Se esperaban {N_RAW_KAGGLE} .raw en {carpeta} y hay {n}.")
    return carpeta


def leer_veredictos(ruta: Path | None = None) -> dict:
    """Lee el JSON de veredictos del hueso experto."""
    ruta = Path(ruta) if ruta is not None else repo_root() / "anotacion/hueso-experto/veredictos-hueso-experto.json"
    return json.loads(ruta.read_text(encoding="utf-8"))


def casos_seleccionados(dataset: Literal["kaggle", "mendeley"], veredictos: dict) -> list[dict]:
    """Casos con veredicto `SOLO_EN_CORTES` o `USABLE`: `n`, `paciente`, `mendeley`, `nucleo`.

    En `mendeley` quedan solo los que tienen archivo Mendeley y se excluye n = 41
    (mismo paciente que n = 4).
    """
    if dataset not in ("kaggle", "mendeley"):
        raise ValueError(f"Dataset desconocido: '{dataset}'. Use 'kaggle' o 'mendeley'.")
    out = []
    for c in veredictos["casos"]:
        if c["veredicto"] not in ("SOLO_EN_CORTES", "USABLE"):
            continue
        if dataset == "mendeley" and (not c.get("mendeley") or c["n"] == 41):
            continue
        out.append({
            "n": c["n"],
            "paciente": c["paciente"],
            "mendeley": c.get("mendeley"),
            "nucleo": list(c["cortes_usables_nucleo"]),
        })
    return out


def _ruta_mendeley(n: int) -> Path:
    equiv = leer_veredictos()["equivalencia_mendeley_kaggle"]
    ks = [k for k, v in equiv.items() if v == f"STIR_{n}"]
    if not ks:
        raise ValueError(f"El caso n = {n} no tiene equivalente en Mendeley según la cabecera del JSON.")
    cand = sorted((repo_root() / "data/raw").glob(f"{ks[0]}_*.raw"))
    if not cand:
        raise FileNotFoundError(f"No hay {ks[0]}_*.raw en data/raw/.")
    return cand[0]


def leer_stir(n: int, dataset: str) -> np.ndarray:
    """Lee la STIR del caso `n` como uint16 (D, 400, 400)."""
    if dataset == "kaggle":
        cand = sorted((asegurar_kaggle() / "STIR").glob(f"STIR_{n}_*.raw"))
        cand = [p for p in cand if parse_raw_filename(p).patient == n]
        if not cand:
            raise FileNotFoundError(f"No hay STIR_{n}_*.raw en Kaggle.")
        ruta = cand[0]
    elif dataset == "mendeley":
        ruta = _ruta_mendeley(n)
    else:
        raise ValueError(f"Dataset desconocido: '{dataset}'. Use 'kaggle' o 'mendeley'.")
    return read_raw(ruta)


def stir_preprocesada(n: int, dataset: str) -> np.ndarray:
    """STIR uint8 (D, 400, 400) con N4 y normalización de la iteración 1 (caché .npz)."""
    cache = repo_root() / f"data/processed/{dataset}/STIR_{n}.npz"
    if cache.is_file():
        with np.load(cache) as f:
            return f["imgs"]
    p = _params_preprocesado()
    vol = leer_stir(n, dataset)
    mask = body_mask(vol, method=p["mask_method"])
    corr, _ = n4_correct(
        vol, mask, spacing_xyz=p["spacing_xyz_mm"],
        shrink_factor=p["n4"]["shrink_factor"], max_iterations=tuple(p["n4"]["max_iterations"]),
    )
    imgs, par = normalize_to_uint8(corr, p["clip"]["low_pct"], p["clip"]["high_pct"], mask=mask)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, imgs=imgs, params=np.array(json.dumps(par)))
    return imgs
