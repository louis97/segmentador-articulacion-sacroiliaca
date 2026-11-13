"""MedSAM2 sobre Apple Silicon (MPS): checkpoint, predictor y segmentación de un volumen.

Parches en tiempo de ejecución (no se toca el código instalado de MedSAM2):

1. Configuración: el paquete instalado no trae los YAML de ``sam2/configs``, así
   que se incluye una copia de ``sam2.1_hiera_t512.yaml`` (Apache 2.0, MedSAM2)
   en ``sij_medsam2/configs`` y se registra con Hydra al construir el modelo.
2. bfloat16: el predictor convierte ``maskmem_features`` a bfloat16 pensando en
   ``torch.autocast("cuda")``. Sin autocast, el modelo en float32 recibiría una
   memoria en bfloat16 y fallaría; se envuelven ``_run_single_frame_inference`` y
   ``_run_memory_encoder`` para devolver esa memoria en float32.
"""
from __future__ import annotations

import hashlib
import os
import warnings
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

from sij_prep.io import repo_root
from sij_prep.model_view import to_model_input

CHECKPOINT_REPO = "wanglab/MedSAM2"
CHECKPOINT_FILE = "MedSAM2_latest.pt"
# Commit de wanglab/MedSAM2 en Hugging Face (última modificación 2025-07-10) y sha256 del archivo
# que se usó para todos los resultados. `_latest` es un nombre móvil: se fija la revisión y se verifica el sha.
CHECKPOINT_REVISION = "e4a6f35edd7e091619cbc0750f462f1574e23955"
CHECKPOINT_SHA256 = "c92743b99f00d078bf32a3afcc38aaa9faf1c1692dffe3eaa7a90938c1991060"
CONFIG_NAME = "sam2.1_hiera_t512"
CONFIG_DIR = Path(__file__).parent / "configs"


def _sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def descargar_checkpoint(dest: Path | None = None) -> Path:
    """Descarga el checkpoint de MedSAM2 (revisión fija) a `dest` (por defecto `data/modelos`); idempotente.

    Verifica el sha256 contra `CHECKPOINT_SHA256` y lanza `RuntimeError` si no coincide.
    """
    from huggingface_hub import hf_hub_download

    dest = (repo_root() / "data/modelos") if dest is None else Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    ruta = dest / CHECKPOINT_FILE
    if not (ruta.is_file() and ruta.stat().st_size > 0):
        hf_hub_download(repo_id=CHECKPOINT_REPO, filename=CHECKPOINT_FILE, revision=CHECKPOINT_REVISION,
                        local_dir=dest)
    sha = _sha256(ruta)
    if sha != CHECKPOINT_SHA256:
        raise RuntimeError(
            f"El checkpoint {ruta} tiene sha256 {sha} y se esperaba {CHECKPOINT_SHA256} "
            f"(revisión {CHECKPOINT_REVISION} de {CHECKPOINT_REPO}). Bórrelo para descargarlo de nuevo; "
            "si el archivo cambió a propósito, actualice CHECKPOINT_REVISION y CHECKPOINT_SHA256."
        )
    return ruta


def elegir_dispositivo(preferido: str = "mps") -> torch.device:
    """Devuelve `mps` si está disponible; si no, `cpu` con un aviso."""
    if preferido == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferido != "cpu":
        warnings.warn(f"El dispositivo '{preferido}' no está disponible; se usa cpu.")
    return torch.device("cpu")


def _a_float32(salida):
    """Convierte `maskmem_features` a float32 en la salida de un paso del predictor."""
    if isinstance(salida, tuple) and isinstance(salida[0], dict):
        d = salida[0]
        if d.get("maskmem_features") is not None:
            d["maskmem_features"] = d["maskmem_features"].float()
    elif isinstance(salida, tuple) and torch.is_tensor(salida[0]):
        salida = (salida[0].float(), *salida[1:])
    return salida


def _parchear_bfloat16(predictor) -> None:
    for nombre in ("_run_single_frame_inference", "_run_memory_encoder"):
        original = getattr(predictor, nombre)

        def envuelto(*a, _orig=original, **k):
            return _a_float32(_orig(*a, **k))

        setattr(predictor, nombre, envuelto)


def construir_predictor(checkpoint: Path, device: torch.device, parchear_bf16: bool = True):
    """Construye el predictor de video de MedSAM2 (versión sobre arrays) en float32."""
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from sam2.build_sam import build_sam2_video_predictor_npz

    GlobalHydra.instance().clear()  # `import sam2` ya inicializó Hydra con su paquete
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.2"):
        predictor = build_sam2_video_predictor_npz(
            CONFIG_NAME, str(checkpoint), device=str(device)
        )
    predictor.float()
    if parchear_bf16:
        _parchear_bfloat16(predictor)
    return predictor


def segmentar_volumen(predictor, frames: np.ndarray, pistas: dict) -> np.ndarray:
    """Segmenta un volumen propagando las pistas hacia adelante y en reversa.

    `frames`: uint8 (D, 400, 400). `pistas`: `{corte: {obj_id: {"caja": (x0,y0,x1,y1)}
    | {"mascara": bool (400,400)}}}` en píxeles de 400. Devuelve uint8 (D, 400, 400)
    con valores en {0, obj_ids}; los solapes se resuelven por el mayor logit.

    Nota (desvío de la spec §5.3): dentro de MedSAM2, `add_new_mask` reescala la máscara de pista a la
    resolución del modelo y los logits de salida vuelven a (H, W) de forma **bilineal**, no por vecino
    más cercano. El efecto es despreciable (Dice en el corte de pista ≥ 0.97) pero no es una copia exacta.
    """
    frames = np.asarray(frames)
    if frames.ndim != 3 or frames.dtype != np.uint8:
        raise ValueError("`frames` debe ser uint8 (D, H, W).")
    if not pistas:
        raise ValueError("Se necesita al menos una pista.")
    D, H, W = frames.shape
    imgs = torch.from_numpy(np.stack([to_model_input(f) for f in frames]))  # (D,3,512,512)
    etiquetas = np.zeros((D, H, W), dtype=np.uint8)
    hecho = np.zeros(D, dtype=bool)

    with torch.inference_mode():
        estado = predictor.init_state(imgs, H, W)
        for corte, por_obj in pistas.items():
            for obj_id, p in por_obj.items():
                if "caja" in p:
                    predictor.add_new_points_or_box(
                        estado, int(corte), int(obj_id), box=np.asarray(p["caja"], np.float32)
                    )
                else:
                    predictor.add_new_mask(
                        estado, int(corte), int(obj_id), torch.from_numpy(np.asarray(p["mascara"], bool))
                    )
        for reverso in (False, True):
            for k, ids, logits in predictor.propagate_in_video(estado, reverse=reverso):
                if hecho[k] and reverso:
                    continue  # el corte de arranque ya se escribió en la pasada hacia adelante
                lg = logits[:, 0].float().cpu().numpy()  # (n_obj, H, W)
                idx = lg.argmax(0)
                mx = lg.max(0)
                ids_arr = np.asarray(ids, dtype=np.uint8)
                etiquetas[k] = np.where(mx > 0, ids_arr[idx], 0)
                hecho[k] = True
        predictor.reset_state(estado)
    return etiquetas
