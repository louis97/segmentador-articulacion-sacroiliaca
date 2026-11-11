"""Descarga verificada (sha256) del conjunto de datos de Mendeley Data."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import requests

MENDELEY_FILES_API = (
    "https://data.mendeley.com/public-api/datasets/425fcppvkn/files"
    "?folder_id=843dd9ad-682f-43b5-9977-aafcddb7634c&version=1"
)
DATASET_URL = "https://data.mendeley.com/datasets/425fcppvkn/1"


@dataclass(frozen=True)
class RemoteFile:
    """Archivo remoto: nombre, tamaño en bytes, URL de descarga y sha256."""

    filename: str
    size: int
    url: str
    sha256: str


@dataclass(frozen=True)
class DownloadResult:
    """Resultado por archivo: si ya estaba presente o fue descargado."""

    filename: str
    status: Literal["presente", "descargado"]
    size: int
    sha256_ok: bool


class DatasetDownloadError(RuntimeError):
    """Error al descargar o verificar el conjunto de datos."""


# Valores tomados de la API pública de Mendeley (versión 1 del conjunto) el 2026-11-10.
# Permiten verificar los archivos locales sin conexión a la red.
KNOWN_FILES: tuple[RemoteFile, ...] = (
    RemoteFile(
        'SIJ_1_400_400_18_2_.raw',
        5760000,
        'https://data.mendeley.com/public-files/datasets/425fcppvkn/files/48ce59ac-4906-48d1-a679-cf50dcf9c721/file_downloaded',
        '09fd393e12c4d3446b44d142dc1ffbed86106a4b900f90fe5098e2dfd5ce4510',
    ),
    RemoteFile(
        'SIJ_2_400_400_24_2_.raw',
        7680000,
        'https://data.mendeley.com/public-files/datasets/425fcppvkn/files/2eb6eb2e-cc01-49c3-a5be-396b159dd7e3/file_downloaded',
        'c82748f2aa2ec90f0e7aa55b4332677b21a7fb341d5d9173e910d245df16de33',
    ),
    RemoteFile(
        'SIJ_3_400_400_18_2_.raw',
        5760000,
        'https://data.mendeley.com/public-files/datasets/425fcppvkn/files/24bd78fa-29fb-4091-b3ca-3587c0e9f86d/file_downloaded',
        'd315eaa493de8515f77c03680bd6ef6330a469f76d285677d97f4ee97ad26afd',
    ),
    RemoteFile(
        'SIJ_4_400_400_20_2_.raw',
        6400000,
        'https://data.mendeley.com/public-files/datasets/425fcppvkn/files/bf6f5b7e-6d09-48fc-bc3c-b69f43f37cff/file_downloaded',
        '7b4bf7870cd0a68392f0cade5033d75094fd88513068c8482f813e63517c6d0e',
    ),
    RemoteFile(
        'SIJ_5_400_400_23_2_.raw',
        7360000,
        'https://data.mendeley.com/public-files/datasets/425fcppvkn/files/bf03095a-30e6-4866-a00e-d98407cdbc1c/file_downloaded',
        '466e8520ea087d2249d41804712629fce3544eea91765ffb2a33f7c87281ea08',
    ),
    RemoteFile(
        'SIJ_6_400_400_20_2_.raw',
        6400000,
        'https://data.mendeley.com/public-files/datasets/425fcppvkn/files/b8347ec9-b7e7-46c8-b25c-080f6bf6c8e6/file_downloaded',
        'b38485d5fdcdfd18a9fd482875f7848b5fdfcbc5ef13e908bedba363f52df1ec',
    ),
    RemoteFile(
        'SIJ_7_400_400_18_2_.raw',
        5760000,
        'https://data.mendeley.com/public-files/datasets/425fcppvkn/files/00a164b9-463a-40df-b233-48fb1d3fd5f7/file_downloaded',
        '2998f27625b500a0bba07e6dcf1d7f464e3a02ad44d06b2ee53567128163503c',
    ),
    RemoteFile(
        'SIJ_8_400_400_18_2_.raw',
        5760000,
        'https://data.mendeley.com/public-files/datasets/425fcppvkn/files/fefd7867-eb4a-46ab-b700-cd0aee1ce6ac/file_downloaded',
        '8a1de0e875927af8cba51df592b8c16b42fcf6f4b50f54727b98c242efcd739b',
    ),
    RemoteFile(
        'SIJ_9_400_400_22_2_.raw',
        7040000,
        'https://data.mendeley.com/public-files/datasets/425fcppvkn/files/f20c583a-cb01-4be8-8de5-43ce08729137/file_downloaded',
        '27b1c3935e8c5a4add080d7dc5b8b730732b8113d2625aac2fd602eb73dd1c99',
    ),
)


def list_remote_files(api_url: str = MENDELEY_FILES_API, session=None) -> list[RemoteFile]:
    """Consulta la API de Mendeley y devuelve los archivos del conjunto de datos.

    No se usa por defecto (``ensure_dataset`` usa ``KNOWN_FILES``); sirve para refrescar ``KNOWN_FILES``.
    """
    session = session or requests.Session()
    resp = session.get(api_url, timeout=60)
    resp.raise_for_status()
    return [
        RemoteFile(
            filename=item["filename"],
            size=int(item["size"]),
            url=item["content_details"]["download_url"],
            sha256=item["content_details"]["sha256_hash"].lower(),
        )
        for item in resp.json()
    ]


def sha256_of(path: Path) -> str:
    """Calcula el sha256 de un archivo, en hexadecimal minúscula."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(rf: RemoteFile, part: Path, session) -> None:
    """Descarga en streaming a `part` y verifica el sha256."""
    h = hashlib.sha256()
    with session.get(rf.url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(part, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                h.update(chunk)
    if h.hexdigest() != rf.sha256.lower():
        raise ValueError("el sha256 no coincide")


def ensure_dataset(
    dest: Path, files: list[RemoteFile] | None = None, session=None, max_retries: int = 3
) -> list[DownloadResult]:
    """Garantiza que los archivos estén en `dest` con sha256 correcto; descarga los que falten."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    session = session or requests.Session()
    if files is None:
        files = list(KNOWN_FILES)

    results = []
    for rf in files:
        final = dest / rf.filename
        if final.exists() and sha256_of(final) == rf.sha256.lower():
            results.append(DownloadResult(rf.filename, "presente", final.stat().st_size, True))
            continue
        part = dest / (rf.filename + ".part")
        last_exc = None
        for attempt in range(max_retries):
            try:
                _download(rf, part, session)
                part.replace(final)
                break
            except Exception as exc:
                last_exc = exc
                part.unlink(missing_ok=True)
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
        else:
            raise DatasetDownloadError(_mensaje(rf.filename, last_exc)) from last_exc
        results.append(DownloadResult(rf.filename, "descargado", final.stat().st_size, True))
    return results


def _mensaje(que: str, exc: Exception) -> str:
    """Arma el mensaje de error en español con instrucciones de descarga manual."""
    return (
        f"No se pudo descargar {que}: {exc}. "
        f"Descargue los archivos manualmente desde {DATASET_URL} "
        "y colóquelos en la carpeta data/raw/ del repositorio."
    )
