"""Máscaras expertas de hueso en la STIR: registro T1 -> STIR portado y reproducible.

Porta el registro validado (supuesto A de geometría, información mutua de Mattes,
rígido -> afín 3D, elección por MI con tolerancia 0.01 entre identidad, rígido y afín,
etiquetas por vecino más cercano). Los parámetros de la transformación elegida se
congelan en `anotacion/hueso-experto/transformaciones.json`; las corridas siguientes
solo remuestrean con ellos.

Convención de ejes: numpy (z, y, x) = (corte, fila, columna); SimpleITK usa (x, y, z).
Las imágenes se construyen con dirección identidad, spacing en mm y origen tal que el
centro geométrico queda en (0, 0, 0). Etiquetas: 1 = ilíaco, 2 = sacro, 3 = ilíaco contralateral.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from sij_prep.io import repo_root

from .datos import asegurar_kaggle, casos_seleccionados, leer_stir, leer_veredictos

STIR_SP = (0.6, 0.6, 3.0)  # (x, y, z) mm
FOV_MM = 240.0
TOL_MI = 0.01
SEMILLA_FIJA = 1234
UMBRAL_DICE = 0.98
SUPUESTO = "A"
_PAT = re.compile(r"^([A-Za-z0-9]+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_\.raw$")
_DZ_INICIOS = (-24.0, -12.0, -6.0, 0.0, 6.0, 12.0, 24.0)


def _ruta_json() -> Path:
    return repo_root() / "anotacion/hueso-experto/transformaciones.json"


def _ruta_aprobada(n: int) -> Path:
    return repo_root() / f"data/external/hueso-experto-aprobado/lab_{n}_elegida.npy"


def _ruta_cache(n: int) -> Path:
    return repo_root() / f"data/processed/hueso_experto/STIR_{n}.npz"


# ---------------------------------------------------------------- lectura de T1 y etiquetas
def _meta(p: Path) -> dict:
    m = _PAT.match(p.name)
    if not m:
        raise ValueError(f"Nombre de archivo no reconocido: {p.name}")
    return dict(n=int(m[2]), H=int(m[3]), W=int(m[4]), D=int(m[5]), b=int(m[6]))


def _leer_raw(p: Path, big_endian: bool) -> np.ndarray:
    """Lee un .raw de 1 o 2 bytes (big-endian si es de 2) como (D, H, W)."""
    m = _meta(p)
    dt = np.dtype(">u2") if m["b"] == 2 and big_endian else (np.dtype("<u2") if m["b"] == 2 else np.dtype("u1"))
    a = np.fromfile(p, dtype=dt)
    if a.size != m["H"] * m["W"] * m["D"]:
        raise ValueError(f"Tamaño inesperado en {p.name}: {a.size}")
    return a.reshape(m["D"], m["H"], m["W"]).astype(np.uint16 if m["b"] == 2 else np.uint8)


def _buscar(carpeta: Path, prefijo: str, n: int) -> Path:
    cand = [p for p in sorted(carpeta.glob(f"{prefijo}_{n}_*.raw")) if _PAT.match(p.name) and _meta(p)["n"] == n]
    if not cand:
        raise FileNotFoundError(f"No hay {prefijo}_{n}_*.raw en {carpeta}")
    return cand[0]


def _cargar_t1_etiquetas(n: int):
    kg = asegurar_kaggle()
    pt, pb = _buscar(kg / "T1", "T1", n), _buscar(kg / "BoneLabels", "BoneLabels", n)
    t1, bl = _leer_raw(pt, True), _leer_raw(pb, True)
    if t1.shape != bl.shape:
        raise ValueError(f"T1 y BoneLabels difieren en n = {n}: {t1.shape} vs {bl.shape}")
    return t1, bl, _meta(pt)


# ---------------------------------------------------------------- geometría e imágenes
def _spacing_t1(meta: dict, d_stir: int) -> tuple:
    """Supuesto A: píxel = 240/H; grosor tal que la pila T1 cubre lo mismo que la STIR."""
    return (FOV_MM / meta["H"], FOV_MM / meta["H"], d_stir * STIR_SP[2] / meta["D"])


def _origen_centrado(size_xyz, spacing_xyz) -> tuple:
    return tuple(float(v) for v in -(np.array(size_xyz, float) - 1) * np.array(spacing_xyz) / 2.0)


def _a_imagen(arr: np.ndarray, spacing_xyz, uint8: bool = False) -> sitk.Image:
    img = sitk.GetImageFromArray(arr if uint8 else arr.astype(np.float32))
    if not uint8:
        img = sitk.Cast(img, sitk.sitkFloat32)
    img.SetSpacing(tuple(float(s) for s in spacing_xyz))
    img.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    img.SetOrigin(_origen_centrado(img.GetSize(), spacing_xyz))
    return img


def _imagen_referencia(d_stir: int, h: int = 400, w: int = 400) -> sitk.Image:
    """Grilla STIR vacía (solo tamaño, spacing y origen): basta para remuestrear."""
    return _a_imagen(np.zeros((d_stir, h, w), np.uint8), STIR_SP, uint8=True)


# ---------------------------------------------------------------- registro
def _grosero(img: sitk.Image, s: float) -> sitk.Image:
    """Versión suavizada y remuestreada en el plano a píxel `s` mm (z se mantiene)."""
    if s <= img.GetSpacing()[0] + 1e-9:
        return img
    g = sitk.SmoothingRecursiveGaussian(img, 0.5 * s)
    sp = (s, s, img.GetSpacing()[2])
    sz = [int(round(img.GetSize()[i] * img.GetSpacing()[i] / sp[i])) for i in range(3)]
    ctr = np.array(img.GetOrigin()) + (np.array(img.GetSize()) - 1) * np.array(img.GetSpacing()) / 2
    org = ctr - (np.array(sz) - 1) * np.array(sp) / 2
    return sitk.Resample(g, sz, sitk.Transform(), sitk.sitkLinear, tuple(org), sp, (1, 0, 0, 0, 1, 0, 0, 0, 1), 0.0)


def _etapa(fijo, movil, tx, tipo: str, iters: int, muestreo: float, semilla: int):
    lr = 0.5 if tipo == "rig" else 0.2
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(32)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(muestreo, semilla)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsRegularStepGradientDescent(learningRate=lr, minStep=1e-5, numberOfIterations=iters,
                                               relaxationFactor=0.5, gradientMagnitudeTolerance=1e-8)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetInitialTransform(tx, inPlace=False)
    out = R.Execute(fijo, movil)
    if out.GetName() == "CompositeTransform":
        out = sitk.CompositeTransform(out).GetBackTransform()
    return sitk.Euler3DTransform(out) if tipo == "rig" else sitk.AffineTransform(out)


def _a_afin(rig: sitk.Euler3DTransform) -> sitk.AffineTransform:
    A = sitk.AffineTransform(3)
    A.SetMatrix(rig.GetMatrix())
    A.SetTranslation(rig.GetTranslation())
    A.SetCenter(rig.GetCenter())
    return A


def _mi(fijo, movil, tx) -> float:
    """MI de Mattes (mayor = mejor) a resolución completa, con todos los vóxeles (determinista)."""
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(32)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(1.0, 777)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetInitialTransform(tx, inPlace=False)
    return -R.MetricEvaluate(fijo, movil)


def _registrar_rigido_afin(stir_img, t1_img, semilla: int):
    """Registro geométrico con multiinicio en z. Devuelve (rígido, afín).

    Rígido a 2.4 mm (7 inicios en z, se sigue con el de mayor MI) y a 1.2 mm;
    afín a 1.2 y 0.6 mm partiendo del rígido. Muestreo 15 %, 250/250/250/200 iteraciones.
    """
    tx = sitk.CenteredTransformInitializer(stir_img, t1_img, sitk.Euler3DTransform(),
                                           sitk.CenteredTransformInitializerFilter.GEOMETRY)
    f24, m24 = _grosero(stir_img, 2.4), _grosero(t1_img, 2.4)
    mejor = None
    for dz in _DZ_INICIOS:
        t = sitk.Euler3DTransform(tx)
        tr = list(t.GetTranslation())
        tr[2] += dz
        t.SetTranslation(tr)
        t = _etapa(f24, m24, t, "rig", 250, 0.15, semilla)
        sc = _mi(f24, m24, t)
        if mejor is None or sc > mejor[0]:
            mejor = (sc, t)
    rig = _etapa(_grosero(stir_img, 1.2), _grosero(t1_img, 1.2), mejor[1], "rig", 250, 0.15, semilla)
    af = _a_afin(rig)
    for s, it in ((1.2, 250), (0.6, 200)):
        af = _etapa(_grosero(stir_img, s), _grosero(t1_img, s), af, "afin", it, 0.15, semilla)
    return rig, af


def _serializar(tx, tipo: str) -> tuple[list, list]:
    if tipo == "identidad":
        return [], []
    return [float(v) for v in tx.GetParameters()], [float(v) for v in tx.GetFixedParameters()]


def _reconstruir(tipo: str, parametros: list, centro: list) -> sitk.Transform:
    """Reconstruye la transformación SimpleITK a partir de los parámetros guardados."""
    if tipo == "identidad":
        return sitk.Transform(3, sitk.sitkIdentity)
    tx = sitk.Euler3DTransform() if tipo == "rigido" else sitk.AffineTransform(3)
    tx.SetFixedParameters(list(centro))
    tx.SetParameters(list(parametros))
    return tx


def registrar_caso(n: int, semilla: int = SEMILLA_FIJA) -> dict:
    """Registra T1 -> STIR del caso `n` y elige la transformación por MI.

    Afín salvo que el rígido supere al afín en más de 0.01, o que la identidad supere a la elegida
    en más de 0.01. Devuelve `tipo`, `parametros`, `centro`, `geometria` y `mi`.
    """
    t1, bl, meta = _cargar_t1_etiquetas(n)
    stir = leer_stir(n, "kaggle")
    d = stir.shape[0]
    if meta["D"] != d:
        raise ValueError(f"n = {n}: D(T1) = {meta['D']} distinto de D(STIR) = {d}; el supuesto A no aplica.")
    sp_t1 = _spacing_t1(meta, d)
    sI, tI = _a_imagen(stir, STIR_SP), _a_imagen(t1, sp_t1)
    ident = sitk.Transform(3, sitk.sitkIdentity)
    mi0 = _mi(sI, tI, ident)
    rig, af = _registrar_rigido_afin(sI, tI, semilla)
    mr, ma = _mi(sI, tI, rig), _mi(sI, tI, af)
    cand = {"afin": (ma, af), "rigido": (mr, rig)}
    el = "rigido" if mr > ma + TOL_MI else "afin"
    if mi0 > cand[el][0] + TOL_MI:
        tipo, tx, mi_el = "identidad", ident, mi0
    else:
        tipo, (mi_el, tx) = el, cand[el]
    parametros, centro = _serializar(tx, tipo)
    return dict(
        tipo=tipo, parametros=parametros, centro=centro, mi=float(mi_el),
        mi_candidatas=dict(identidad=float(mi0), rigido=float(mr), afin=float(ma)),
        geometria=dict(
            supuesto=SUPUESTO, d_stir=int(d), tam_stir_hw=[int(stir.shape[1]), int(stir.shape[2])],
            spacing_t1=[float(s) for s in sp_t1], origen_t1=list(_origen_centrado(
                (meta["W"], meta["H"], meta["D"]), sp_t1)),
            spacing_stir=list(STIR_SP), origen_stir=list(_origen_centrado(
                (stir.shape[2], stir.shape[1], d), STIR_SP)),
            t1_hwd=[meta["H"], meta["W"], meta["D"]], bytes_t1=meta["b"],
        ),
    )


# ---------------------------------------------------------------- remuestreo y máscara experta
def etiquetas_desde_transformacion(n: int, trans: dict) -> np.ndarray:
    """Lleva BoneLabels de la grilla T1 a la grilla STIR con la transformación guardada (sin optimizar).

    Devuelve uint8 (D, 400, 400), vecino más cercano.
    """
    _, bl, meta = _cargar_t1_etiquetas(n)
    g = trans["geometria"]
    d = g["d_stir"]
    h, w = g.get("tam_stir_hw", [400, 400])
    bI = _a_imagen(bl, g["spacing_t1"], uint8=True)
    ref = _imagen_referencia(d, h, w)
    tx = _reconstruir(trans["tipo"], trans["parametros"], trans["centro"])
    out = sitk.Resample(bI, ref, tx, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(out)


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    s = int(a.sum()) + int(b.sum())
    return float("nan") if s == 0 else float(2.0 * np.logical_and(a, b).sum() / s)


def _dice_por_hueso(gen: np.ndarray, ref: np.ndarray, cortes: list[int]) -> dict:
    """Dice por etiqueta (1 ilíaco, 2 sacro, 3 ilíaco contralateral) en los cortes del núcleo."""
    idx = list(cortes)
    return {str(e): _dice(gen[idx] == e, ref[idx] == e) for e in (1, 2, 3)}


def _pasa(dice: dict) -> bool:
    return all(np.isnan(v) or v >= UMBRAL_DICE for v in dice.values())


def _huella(entrada: dict | None) -> str:
    """SHA-256 del JSON canónico de la entrada del caso (vacío si no hay entrada)."""
    canon = json.dumps(entrada, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def preparar_transformaciones(casos: list[int]) -> Path:
    """Registra cada caso, lo compara con la máscara aprobada y congela `transformaciones.json`.

    Exige Dice >= 0.98 por hueso en los cortes del núcleo. Si no pasa, reintenta con un solo hilo y
    semilla fija; si sigue sin pasar, marca `"fuente": "aprobada"` (se usa la máscara aprobada).
    El reintento usa la misma semilla (`SEMILLA_FIJA`): solo cambia el número de hilos.
    Cada entrada registra `semilla` e `hilos` con los que se congeló.
    Conserva las entradas de otros casos ya presentes en el archivo.
    """
    nucleo = {c["n"]: c["nucleo"] for c in casos_seleccionados("kaggle", leer_veredictos())}
    ruta = _ruta_json()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    previo = {e["n"]: e for e in json.loads(ruta.read_text(encoding="utf-8"))["casos"]} if ruta.is_file() else {}
    hilos = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
    for n in casos:
        if n not in nucleo:
            raise ValueError(f"El caso n = {n} no está entre los seleccionados de kaggle.")
        ref = np.load(_ruta_aprobada(n))
        entrada = None
        for intento in (1, 2):
            if intento == 2:
                sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
            n_hilos = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
            try:
                tr = registrar_caso(n, SEMILLA_FIJA)
                gen = etiquetas_desde_transformacion(n, tr)
            finally:
                sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(hilos)
            dice = _dice_por_hueso(gen, ref, nucleo[n])
            entrada = dict(n=n, **tr, dice_vs_aprobada=dice, intentos=intento,
                           semilla=SEMILLA_FIJA, hilos=int(n_hilos), fecha=date.today().isoformat())
            if _pasa(dice):
                entrada["fuente"] = "registro"
                break
            entrada["fuente"] = "aprobada"
        print(f"n = {n}: {entrada['tipo']} MI {entrada['mi']:.3f} Dice {entrada['dice_vs_aprobada']} "
              f"fuente {entrada['fuente']} (intentos {entrada['intentos']})", flush=True)
        previo[n] = entrada
        doc = dict(version=1, descripcion="Transformaciones T1 -> STIR congeladas de las máscaras expertas de hueso.",
                   casos=[previo[k] for k in sorted(previo)])
        ruta.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return ruta


def _guardar_cache(n: int, etiquetas: np.ndarray, fuente: str, huella: str) -> None:
    ruta = _ruta_cache(n)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ruta, etiquetas=etiquetas.astype(np.uint8), fuente=np.array(fuente), huella=np.array(huella))


def mascara_experta(n: int, ruta_json: Path | None = None) -> np.ndarray:
    """Máscara experta de hueso del caso `n` como uint8 (D, 400, 400), etiquetas 0-3.

    Orden: 1) caché `data/processed/hueso_experto/STIR_{n}.npz`, solo si su `fuente` y su huella
    (SHA-256 de la entrada del caso en `transformaciones.json`) coinciden con las actuales; si faltan
    o difieren se regenera y se sobrescribe; 2) `transformaciones.json` con fuente "registro"
    (solo remuestrea); 3) la máscara aprobada (fuente "aprobada").
    `ruta_json` permite apuntar a otra copia del JSON (por defecto, la del repo).
    """
    ruta = Path(ruta_json) if ruta_json is not None else _ruta_json()
    entrada = None
    if ruta.is_file():
        entrada = next((e for e in json.loads(ruta.read_text(encoding="utf-8"))["casos"] if e["n"] == n), None)
    fuente = "registro" if entrada is not None and entrada["fuente"] == "registro" else "aprobada"
    huella = _huella(entrada)
    cache = _ruta_cache(n)
    if cache.is_file():
        with np.load(cache) as f:
            if "huella" in f.files and str(f["huella"]) == huella and str(f["fuente"]) == fuente:
                return f["etiquetas"]
    if fuente == "registro":
        et = etiquetas_desde_transformacion(n, entrada)
    else:
        ap = _ruta_aprobada(n)
        if not ap.is_file():
            raise FileNotFoundError(f"No hay transformación congelada ni máscara aprobada para n = {n} ({ap}).")
        et = np.load(ap).astype(np.uint8)
    _guardar_cache(n, et, fuente, huella)
    return et
