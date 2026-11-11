"""Entrada del modelo (réplica de lo que hace MedSAM2) y sugerencia de corte clave."""
from __future__ import annotations

import numpy as np
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MODEL_IMAGE_SIZE = 512

_MEAN = np.asarray(IMAGENET_MEAN, dtype=np.float32)
_STD = np.asarray(IMAGENET_STD, dtype=np.float32)


def to_model_input(slice_u8: np.ndarray) -> np.ndarray:
    """Convierte un corte uint8 (H,W) en la entrada de MedSAM2: float32 (3,512,512).

    Replica el corte en 3 canales, lo redimensiona a 512x512 (filtro por defecto
    de Pillow, sin conservar la relación de aspecto), divide entre 255 y aplica
    la normalización de ImageNet por canal.
    """
    img = Image.fromarray(slice_u8).convert("RGB").resize((MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE))
    x = np.asarray(img, dtype=np.float32) / 255.0  # (512, 512, 3)
    x = (x - _MEAN) / _STD
    return np.ascontiguousarray(x.transpose(2, 0, 1), dtype=np.float32)


def denormalize_for_display(x: np.ndarray) -> np.ndarray:
    """Deshace la normalización de ImageNet: (3,512,512) -> (512,512,3) en [0,1], recortado."""
    y = np.asarray(x, dtype=np.float32) * _STD[:, None, None] + _MEAN[:, None, None]
    return np.clip(y.transpose(1, 2, 0), 0.0, 1.0)


def suggest_key_slice(
    vol_u8: np.ndarray, mask: np.ndarray | None = None
) -> tuple[int, np.ndarray]:
    """Devuelve `(índice, perfil_por_corte)`: el corte clave sugerido y un perfil descriptivo.

    La sugerencia es siempre el corte central `D // 2`, por convención; se
    justifica con la comparación visual en los 9 volúmenes, donde el centro
    mostró el sacro con sus agujeros y ambas articulaciones. El perfil NO decide
    la sugerencia: es descriptivo y sirve para ver cómo cambian la cobertura y el
    contenido de anterior a posterior. Con `mask` es la fracción de píxeles de
    cada corte dentro del cuerpo; sin `mask`, la intensidad media del corte.
    Tiene forma (D,), `float32` y es finito. La caja final la dibuja una persona.
    """
    vol_u8 = np.asarray(vol_u8)
    if vol_u8.ndim != 3:
        raise ValueError("Se esperaba un volumen 3D (D, H, W).")
    if mask is None:
        profile = vol_u8.mean(axis=(1, 2))
    else:
        m = np.asarray(mask, dtype=bool)
        if m.shape != vol_u8.shape:
            raise ValueError("La máscara debe tener la misma forma que el volumen.")
        profile = m.mean(axis=(1, 2))
    return vol_u8.shape[0] // 2, profile.astype(np.float32)
