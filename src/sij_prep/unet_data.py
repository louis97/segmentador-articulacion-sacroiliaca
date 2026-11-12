"""Preparación 2: del volumen procesado + máscara al dataset de la U-Net (2D y 2.5D).

Entrada: los volúmenes `uint8` (D, 400, 400) de `data/processed/` (notebook 01) y
las máscaras binarias de la articulación (D, 400, 400) que salen de MedSAM2 y de
la corrección manual.

Todo lo que transforma la imagen (normalización, relleno, apilado 2.5D) vive aquí
para que el entrenamiento, la evaluación y la aplicación usen exactamente el mismo
código: si la app preparara la imagen distinto que el entrenamiento, el modelo
vería datos que nunca vio.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

from sij_prep.intensity import body_mask
from sij_prep.io import load_nifti, load_npz

# La U-Net reduce la imagen a la mitad en cada nivel del codificador. Con 5 niveles
# (lo habitual con codificadores tipo ResNet) el lado debe ser múltiplo de 2**5 = 32.
# 400 no lo es (400/32 = 12.5); el múltiplo siguiente es 416.
UNET_DIVISOR = 32
TARGET_SIZE = 416


# ---------------------------------------------------------------------------
# Máscaras
# ---------------------------------------------------------------------------

def load_mask(path: Path | str) -> np.ndarray:
    """Carga una máscara (`.npz` con la clave `mask`, o `.nii.gz`) como `uint8` 0/1.

    Acepta valores {0, 1} o {0, 255} (algunas herramientas guardan 255). Cualquier
    otro valor detiene el proceso: significaría varias clases o una máscara no
    binaria, y eso hay que decidirlo explícitamente, no adivinarlo.
    """
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path) as f:
            if "mask" not in f:
                raise KeyError(f"'{path.name}' no tiene la clave 'mask' (tiene {list(f.keys())}).")
            m = f["mask"]
    elif path.name.endswith(".nii.gz") or path.suffix == ".nii":
        m, _ = load_nifti(path)
    else:
        raise ValueError(f"Formato de máscara no soportado: '{path.name}'. Use .npz o .nii.gz.")

    values = set(np.unique(m).tolist())
    if values <= {0, 1}:
        return m.astype(np.uint8)
    if values <= {0, 255}:
        return (m > 0).astype(np.uint8)
    raise ValueError(
        f"La máscara '{path.name}' tiene valores {sorted(values)[:10]}; se esperaba "
        "binaria ({0,1} o {0,255}). Si hay varias clases (p. ej. izquierda/derecha), "
        "hay que adaptar la preparación."
    )


def find_mask_file(mask_dir: Path, stem: str) -> Path | None:
    """Busca `<stem>.npz` o `<stem>.nii.gz` en `mask_dir`."""
    for name in (f"{stem}.npz", f"{stem}.nii.gz"):
        p = Path(mask_dir) / name
        if p.is_file():
            return p
    return None


def make_placeholder_mask(shape: tuple[int, int, int]) -> np.ndarray:
    """Máscara SINTÉTICA para probar el código mientras no existan las reales.

    Dibuja dos franjas elípticas finas y simétricas a ambos lados del centro en los
    cortes centrales. NO es anatomía: su única función es que el resto del flujo
    (apilado, particiones, aumento, guardado) se pueda ejecutar de punta a punta.
    """
    d, h, w = shape
    m = np.zeros(shape, dtype=np.uint8)
    rr, cc = np.mgrid[0:h, 0:w]
    k0, k1 = int(round(d * 0.25)), int(round(d * 0.75))
    for side in (-1, 1):
        cy, cx = h * 0.55, w / 2 + side * w * 0.15
        # elipse alargada en vertical (como el trazo de la articulación en coronal)
        ell = ((rr - cy) / (h * 0.14)) ** 2 + ((cc - cx) / (w * 0.012)) ** 2 <= 1.0
        m[k0:k1, ell] = 1
    return m


# ---------------------------------------------------------------------------
# Intensidad y tamaño
# ---------------------------------------------------------------------------

def zscore_volume(vol_u8: np.ndarray, body: np.ndarray) -> tuple[np.ndarray, dict]:
    """Z-score por volumen con la media y la desviación DENTRO del cuerpo.

    Se usa el cuerpo y no todo el volumen porque el fondo (aire, ~0) ocupa entre el
    8 % y el 26 % del volumen según el paciente; incluirlo cambiaría la media y la
    desviación por cuánto aire hay, no por el tejido. Se calcula por volumen, nunca
    por corte, para que el mismo tejido tenga el mismo valor en cortes vecinos
    (requisito de la U-Net 2.5D, que compara cortes entre sí).
    """
    v = np.asarray(vol_u8, dtype=np.float32)
    inside = v[np.asarray(body, dtype=bool)]
    mean, std = float(inside.mean()), float(inside.std())
    if std < 1e-6:
        raise ValueError("Desviación ~0 dentro del cuerpo: el volumen parece vacío o constante.")
    return (v - mean) / std, {"mean": mean, "std": std}


def pad_to_multiple(
    vol: np.ndarray, target: int = TARGET_SIZE, value: float = 0.0
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Rellena (D,H,W) en H y W hasta `target`, centrado. Devuelve también el relleno.

    El relleno es `(arriba, abajo, izquierda, derecha)` y sirve para deshacerlo con
    `unpad` (la app debe devolver la máscara al tamaño original 400x400).
    """
    _, h, w = vol.shape
    if h > target or w > target:
        raise ValueError(f"El corte {h}x{w} es mayor que el tamaño objetivo {target}.")
    top, left = (target - h) // 2, (target - w) // 2
    bottom, right = target - h - top, target - w - left
    out = np.pad(vol, ((0, 0), (top, bottom), (left, right)), constant_values=value)
    return out, (top, bottom, left, right)


def unpad(arr: np.ndarray, pad: tuple[int, int, int, int]) -> np.ndarray:
    """Deshace `pad_to_multiple` sobre las dos últimas dimensiones."""
    top, bottom, left, right = pad
    h, w = arr.shape[-2:]
    return arr[..., top : h - bottom, left : w - right]


def prepare_volume(vol_u8: np.ndarray, target: int = TARGET_SIZE) -> tuple[np.ndarray, np.ndarray, dict]:
    """Imagen lista para la U-Net: máscara del cuerpo + z-score + relleno.

    Es la misma función que debe usar la aplicación con un volumen nuevo (ya
    pasado por el notebook 01). El relleno toma el valor del fondo normalizado
    (lo que vale una intensidad 0 tras el z-score), así el borde añadido se ve
    como aire y no como tejido gris.
    """
    body = body_mask(vol_u8)
    z, stats = zscore_volume(vol_u8, body)
    background = (0.0 - stats["mean"]) / stats["std"]
    z_pad, pad = pad_to_multiple(z, target, value=background)
    body_pad, _ = pad_to_multiple(body.astype(np.uint8), target, value=0)
    stats.update({"pad": list(pad), "background_value": background})
    return z_pad.astype(np.float32), body_pad, stats


# ---------------------------------------------------------------------------
# 2D y 2.5D
# ---------------------------------------------------------------------------

def stack_25d(vol: np.ndarray, k: int, context: int = 1) -> np.ndarray:
    """Canales `[k-context, ..., k, ..., k+context]` de un volumen (D,H,W).

    En los extremos se repite el corte del borde (el corte 0 usa [0, 0, 1]). Se
    repite en lugar de rellenar con ceros porque un canal negro sería una imagen
    falsa que el modelo no vería en ningún otro corte.
    """
    d = vol.shape[0]
    idx = np.clip(np.arange(k - context, k + context + 1), 0, d - 1)
    return vol[idx]


def slice_input(vol: np.ndarray, k: int, mode: str) -> np.ndarray:
    """Entrada de la red para el corte `k`: (1,H,W) en '2d' o (3,H,W) en '2.5d'."""
    if mode == "2d":
        return vol[k : k + 1]
    if mode == "2.5d":
        return stack_25d(vol, k, context=1)
    raise ValueError(f"Modo desconocido '{mode}'. Use '2d' o '2.5d'.")


# ---------------------------------------------------------------------------
# Particiones por paciente
# ---------------------------------------------------------------------------

def lopo_splits(patients: list[str]) -> list[dict]:
    """Leave-one-patient-out con un paciente de validación por partición.

    Partición i: prueba = paciente i, validación = paciente i+1 (circular),
    entrenamiento = los demás. Así cada paciente es prueba exactamente una vez y
    validación exactamente una vez, y ningún corte de un paciente aparece en dos
    conjuntos de la misma partición.
    """
    n = len(patients)
    if n < 3:
        raise ValueError("Se necesitan al menos 3 pacientes (entrenamiento, validación y prueba).")
    folds = []
    for i in range(n):
        test, val = patients[i], patients[(i + 1) % n]
        train = [p for p in patients if p not in (test, val)]
        folds.append({"fold": i, "train": train, "val": [val], "test": [test]})
    return folds


def check_splits(folds: list[dict], patients: list[str]) -> None:
    """Falla si hay fuga entre conjuntos o si algún paciente no se usa como prueba."""
    for f in folds:
        tr, va, te = set(f["train"]), set(f["val"]), set(f["test"])
        if tr & va or tr & te or va & te:
            raise AssertionError(f"Fuga de pacientes en la partición {f['fold']}.")
        if tr | va | te != set(patients):
            raise AssertionError(f"La partición {f['fold']} no cubre a todos los pacientes.")
    tested = sorted(p for f in folds for p in f["test"])
    if tested != sorted(patients):
        raise AssertionError("Cada paciente debe ser prueba exactamente una vez.")


# ---------------------------------------------------------------------------
# Aumento de datos
# ---------------------------------------------------------------------------

@dataclass
class AugmentConfig:
    """Rangos del aumento de datos (ver la justificación en el notebook 02)."""

    p_hflip: float = 0.5
    max_rotation_deg: float = 10.0
    scale_range: tuple[float, float] = (0.9, 1.1)
    max_shift_px: float = 20.0
    gamma_range: tuple[float, float] = (0.8, 1.25)
    contrast_range: tuple[float, float] = (0.9, 1.1)
    brightness_shift: float = 0.1
    noise_std: float = 0.05


def augment(
    x: np.ndarray, y: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig = AugmentConfig()
) -> tuple[np.ndarray, np.ndarray]:
    """Aumento sobre una muestra: `x` (C,H,W) float32 en z-score, `y` (H,W) 0/1.

    La misma transformación geométrica se aplica a todos los canales y a la
    máscara (si no, la máscara dejaría de coincidir con la imagen, y en 2.5D los
    cortes vecinos dejarían de estar alineados). La imagen se interpola de forma
    lineal y la máscara por vecino más cercano, para que siga siendo 0/1.
    """
    c, h, w = x.shape
    bg = float(x.min())  # valor del aire: se usa para rellenar lo que entra por el borde

    if rng.random() < cfg.p_hflip:
        x, y = x[:, :, ::-1], y[:, ::-1]

    ang = np.deg2rad(rng.uniform(-cfg.max_rotation_deg, cfg.max_rotation_deg))
    s = rng.uniform(*cfg.scale_range)
    ty, tx = rng.uniform(-cfg.max_shift_px, cfg.max_shift_px, size=2)
    # matriz que lleva coordenadas de salida a coordenadas de entrada (rotación + escala)
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) / s
    center = np.array([(h - 1) / 2, (w - 1) / 2])
    offset = center - rot @ (center + np.array([ty, tx]))

    x_out = np.empty_like(x, dtype=np.float32)
    for ch in range(c):
        x_out[ch] = ndimage.affine_transform(
            np.ascontiguousarray(x[ch]), rot, offset=offset, order=1, mode="constant", cval=bg
        )
    y_out = ndimage.affine_transform(
        np.ascontiguousarray(y).astype(np.float32), rot, offset=offset, order=0, mode="constant", cval=0
    )

    # Intensidad: la misma para todos los canales (vienen del mismo volumen y de la
    # misma normalización; cambiarlos por separado rompería la coherencia 2.5D).
    lo, hi = x_out.min(), x_out.max()
    u = (x_out - lo) / max(hi - lo, 1e-6)  # gamma se define sobre [0,1]
    u = u ** rng.uniform(*cfg.gamma_range)
    x_out = u * (hi - lo) + lo
    x_out = x_out * rng.uniform(*cfg.contrast_range) + rng.uniform(-cfg.brightness_shift, cfg.brightness_shift)
    x_out = x_out + rng.normal(0.0, cfg.noise_std, size=x_out.shape).astype(np.float32)

    return x_out.astype(np.float32), (y_out > 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# Dataset por cortes (compatible con torch.utils.data.DataLoader)
# ---------------------------------------------------------------------------

@dataclass
class SliceDataset:
    """Dataset de cortes para entrenar la U-Net, sin depender de PyTorch.

    `volumes` es un diccionario `{paciente: (imagen (D,H,W) float32, máscara (D,H,W) uint8)}`
    con los arreglos de `data/unet/`. Cada elemento devuelve `(x, y)`:
    `x` (C,H,W) float32 con C=1 (2d) o C=3 (2.5d) e `y` (1,H,W) float32 en {0,1}.

    Tiene `__len__` y `__getitem__`, así que `torch.utils.data.DataLoader` lo
    acepta directamente y convierte los arreglos en tensores. Con `num_workers>0`
    hay que reiniciar `rng` en cada proceso (`worker_init_fn`), si no todos los
    procesos generan el mismo aumento; con un dataset tan pequeño basta `num_workers=0`.
    """

    volumes: dict[str, tuple[np.ndarray, np.ndarray]]
    patients: list[str]
    mode: str = "2.5d"
    augment: bool = False
    aug_cfg: AugmentConfig = field(default_factory=AugmentConfig)
    seed: int = 0

    def __post_init__(self) -> None:
        if self.mode not in ("2d", "2.5d"):
            raise ValueError(f"Modo desconocido '{self.mode}'.")
        self.index = [(p, k) for p in self.patients for k in range(self.volumes[p][0].shape[0])]
        self.rng = np.random.default_rng(self.seed)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        p, k = self.index[i]
        img, msk = self.volumes[p]
        x = slice_input(img, k, self.mode).astype(np.float32)
        y = msk[k]
        if self.augment:
            x, y = augment(x, y, self.rng, self.aug_cfg)
        return np.ascontiguousarray(x), np.ascontiguousarray(y[None].astype(np.float32))

    def positive_flags(self) -> np.ndarray:
        """`True` para los cortes que contienen articulación (útil para muestrear)."""
        return np.array([self.volumes[p][1][k].any() for p, k in self.index])


def load_unet_volumes(unet_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Carga `data/unet/SIJ_<n>.npz` como `{paciente: (imagen, máscara)}`, en orden numérico."""
    files = sorted(Path(unet_dir).glob("SIJ_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    out = {}
    for f in files:
        with np.load(f) as z:
            out[f.stem] = (z["image"], z["mask"])
    return out


__all__ = [
    "AugmentConfig", "SliceDataset", "TARGET_SIZE", "augment", "check_splits", "find_mask_file",
    "load_mask", "load_npz", "load_unet_volumes", "lopo_splits", "make_placeholder_mask",
    "pad_to_multiple", "prepare_volume", "slice_input", "stack_25d", "unpad", "zscore_volume",
]
