"""Pipeline de punta a punta: datos -> máscara experta -> pistas -> MedSAM2 -> métricas -> CSV.

Escribe en `resultados/medsam2/<dataset>/` (CSV, JSON) y las predicciones en
`data/processed/medsam2/<dataset>/STIR_{n}_{tipo}.npz` (caché, ignorada por git).
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata as md
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np

from sij_prep.io import repo_root

from .datos import casos_seleccionados, leer_veredictos, stir_preprocesada
from .metricas import metricas_corte
from .modelo import (CHECKPOINT_FILE, _sha256, construir_predictor, descargar_checkpoint,
                     elegir_dispositivo, segmentar_volumen)
from .pistas import MIN_CORTES_EVAL, TIPOS_PISTA, construir_pistas, cortes_evaluacion, cortes_pista
from .registro import mascara_experta

NAN = float("nan")
PX_MM = 0.6  # desvío de spec §5.5: el eje también se informa en mm
ESTRUCTURAS = ("iliaco", "sacro", "iliaco_contra", "franja_A", "franja_B")
METRICAS = ("dice", "hd95_px", "hd95_mm", "dist_media_px", "eje_mediana_px", "eje_p95_px",
            "eje_mediana_mm", "eje_p95_mm", "frac_filas_discordantes")
COLS_CORTE = ["n", "paciente", "tipo", "corte", "distancia_a_pista", "estructura", *METRICAS,
              "vacio_pred", "vacio_exp"]
_HUESOS = {"iliaco": 1, "sacro": 2, "iliaco_contra": 3}


def dir_resultados(dataset: str) -> Path:
    return repo_root() / "resultados/medsam2" / dataset


def _ruta_pred(dataset: str, n: int, tipo: str) -> Path:
    return repo_root() / f"data/processed/medsam2/{dataset}/STIR_{n}_{tipo}.npz"


def _num(x):
    """Valor serializable en CSV: NaN -> 'nan', bool -> 0/1."""
    if isinstance(x, (bool, np.bool_)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return "nan" if math.isnan(x) else f"{float(x):.6g}"
    return x


def _escribir_csv(ruta: Path, columnas: list[str], filas: list[dict]) -> None:
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columnas)
        w.writeheader()
        for fila in filas:
            w.writerow({c: _num(fila.get(c, "")) for c in columnas})


def _media(v) -> float:
    v = [x for x in v if not math.isnan(x)]
    return float(np.mean(v)) if v else NAN


def _json_ok(o):
    """Convierte NaN a None y tipos numpy a nativos para JSON."""
    if isinstance(o, dict):
        return {str(k): _json_ok(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_ok(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if math.isnan(o) else float(o)
    return o


_CAMPOS_CACHE = ("pred", "dispositivo", "segundos", "checkpoint_sha256", "huella_pistas")


def _huella_pistas(tipo: str, cortes: list[int], pistas: dict) -> str:
    """SHA-256 de lo que recibe el modelo: tipo, cortes de pista y, por (corte, objeto), la caja o la máscara."""
    h = hashlib.sha256()
    h.update(f"{tipo}|{sorted(int(c) for c in cortes)}".encode())
    for corte in sorted(pistas):
        for obj in sorted(pistas[corte]):
            p = pistas[corte][obj]
            h.update(f"|{int(corte)}:{int(obj)}:".encode())
            if "caja" in p:
                h.update(b"caja" + np.asarray(p["caja"], np.float32).tobytes())
            else:
                h.update(b"mascara" + np.packbits(np.asarray(p["mascara"], bool)).tobytes())
    return h.hexdigest()


def _leer_cache(ruta: Path, huella: str) -> dict | None:
    """Lee una predicción cacheada con su procedencia.

    None si falta, es antigua (sin procedencia o sin huella de pistas) o la huella no coincide con las
    pistas de hoy (las pistas cambiaron desde que se calculó la predicción).
    """
    if not ruta.is_file():
        return None
    with np.load(ruta) as f:
        if not all(k in f.files for k in _CAMPOS_CACHE) or str(f["huella_pistas"]) != huella:
            return None
        return dict(pred=f["pred"], dispositivo=str(f["dispositivo"]), segundos=float(f["segundos"]),
                    checkpoint_sha256=str(f["checkpoint_sha256"]))


def migrar_huellas_cache(dataset: Literal["kaggle", "mendeley"], tipos: tuple = TIPOS_PISTA) -> dict:
    """Migración única: añade `huella_pistas` a las `.npz` que no la tienen, SIN recalcular predicciones.

    Solo se añade si las pistas que se reconstruyen hoy (cortes de pista y objetos con pista) coinciden
    con las de `configuracion.json` de ese dataset, que es lo que se usó al calcular la predicción;
    si no coinciden, la `.npz` se deja sin tocar (y `ejecutar_pipeline` la recalculará). La comparación
    no puede ver las máscaras mismas (el JSON no las guarda); eso lo cubre `transformaciones.json`.
    Devuelve `{"migradas": [...], "ya_tenian": [...], "no_coinciden": [...]}` con pares `(n, tipo)`.
    """
    cfg = json.loads((dir_resultados(dataset) / "configuracion.json").read_text(encoding="utf-8"))
    res = {"migradas": [], "ya_tenian": [], "no_coinciden": []}
    for caso in casos_seleccionados(dataset, leer_veredictos()):
        n, nucleo = caso["n"], sorted(caso["nucleo"])
        exp = None
        for tipo in tipos:
            ruta = _ruta_pred(dataset, n, tipo)
            if not ruta.is_file():
                continue
            with np.load(ruta) as f:
                datos = {k: f[k] for k in f.files}
            if "huella_pistas" in datos:
                res["ya_tenian"].append((n, tipo))
                continue
            exp = mascara_experta(n) if exp is None else exp
            pistas_c = cortes_pista(nucleo, tipo)
            pistas = construir_pistas(exp, pistas_c, tipo)
            c = cfg["casos"].get(str(n), {}).get("tipos", {}).get(tipo, {})
            objetos = sorted({o for por in pistas.values() for o in por})
            if c.get("cortes_pista") != pistas_c or c.get("objetos_con_pista") != objetos:
                res["no_coinciden"].append((n, tipo))
                continue
            datos["huella_pistas"] = np.array(_huella_pistas(tipo, pistas_c, pistas))
            np.savez_compressed(ruta, **datos)
            res["migradas"].append((n, tipo))
    return res


def _dice(a, b) -> float:
    s = a.sum() + b.sum()
    return NAN if s == 0 else float(2 * (a & b).sum() / s)


def _por_caso(filas_corte: list[dict]) -> list[dict]:
    """Promedio por (n, tipo, estructura) sobre cortes, excluyendo NaN; cuenta vacíos."""
    grupos: dict = {}
    for f in filas_corte:
        grupos.setdefault((f["n"], f["paciente"], f["tipo"], f["estructura"]), []).append(f)
    out = []
    for (n, pac, tipo, est), fs in grupos.items():
        fila = dict(n=n, paciente=pac, tipo=tipo, estructura=est, n_cortes=len(fs),
                    n_vacio_pred=sum(int(f["vacio_pred"]) for f in fs),
                    n_vacio_exp=sum(int(f["vacio_exp"]) for f in fs))
        for m in METRICAS:
            fila[m] = _media([f[m] for f in fs])
        out.append(fila)
    return out


def _por_paciente(filas_caso: list[dict]) -> list[dict]:
    """Promedio de los casos de un mismo paciente (n = 4 y 41 juntos), por (paciente, tipo, estructura)."""
    grupos: dict = {}
    for f in filas_caso:
        grupos.setdefault((f["paciente"], f["tipo"], f["estructura"]), []).append(f)
    out = []
    for (pac, tipo, est), fs in grupos.items():
        fila = dict(paciente=pac, tipo=tipo, estructura=est,
                    n_vacio_pred=sum(f["n_vacio_pred"] for f in fs),
                    n_vacio_exp=sum(f["n_vacio_exp"] for f in fs))
        for m in METRICAS:
            fila[m] = _media([f[m] for f in fs])
        out.append(fila)
    return out


def _estad(v: list[float]) -> dict:
    v = np.array([x for x in v if not math.isnan(x)])
    if v.size == 0:
        return dict(n=0, media=NAN, desv=NAN, mediana=NAN, p25=NAN, p75=NAN)
    return dict(n=int(v.size), media=float(v.mean()), desv=float(v.std(ddof=1)) if v.size > 1 else NAN,
                mediana=float(np.median(v)), p25=float(np.percentile(v, 25)), p75=float(np.percentile(v, 75)))


COLS_RESUMEN = ["conjunto", "tipo", "estructura", "n_pacientes", "metrica", "n", "media", "desv",
                "mediana", "p25", "p75", "n_vacio_pred", "n_vacio_exp"]


def resumir(filas_corte: list[dict], tipos: tuple) -> list[dict]:
    """Resumen entre pacientes por tipo y estructura: 'todos' y 'comun' (pacientes con métricas en todos los tipos)."""
    pac = _por_paciente(_por_caso(filas_corte))
    tipos_pres = [t for t in tipos if any(p["tipo"] == t for p in pac)]
    por_tipo = {t: {p["paciente"] for p in pac if p["tipo"] == t} for t in tipos_pres}
    comun = set.intersection(*por_tipo.values()) if por_tipo else set()
    out = []
    for conjunto, filtro in (("todos", None), ("comun", comun)):
        for t in tipos_pres:
            for est in ESTRUCTURAS:
                fs = [p for p in pac if p["tipo"] == t and p["estructura"] == est
                      and (filtro is None or p["paciente"] in filtro)]
                for m in METRICAS:
                    e = _estad([p[m] for p in fs])
                    out.append(dict(conjunto=conjunto, tipo=t, estructura=est, n_pacientes=len(fs),
                                    metrica=m, **e,
                                    n_vacio_pred=sum(p["n_vacio_pred"] for p in fs),
                                    n_vacio_exp=sum(p["n_vacio_exp"] for p in fs)))
    return out


def _versiones() -> dict:
    v = {}
    for p in ("torch", "numpy", "SimpleITK", "MedSAM2", "matplotlib", "hydra-core"):
        try:
            v[p] = md.version(p)
        except md.PackageNotFoundError:
            v[p] = None
    return v


def ejecutar_pipeline(dataset: Literal["kaggle", "mendeley"] = "kaggle", dispositivo: str = "mps",
                      tipos: tuple = TIPOS_PISTA, rehacer: bool = False) -> dict:
    """Corre el pipeline completo para `dataset` y devuelve la configuración escrita.

    Construye el predictor una sola vez. Las predicciones se cachean por (caso, tipo) con su
    procedencia (dispositivo, segundos del modelo, sha256 del checkpoint) y una huella de las pistas
    (tipo, cortes y cajas o máscaras); una caché sin esos campos, o cuya huella no coincide con las
    pistas de hoy, se considera inválida y se recalcula (las `.npz` antiguas sin huella se migran con
    `migrar_huellas_cache`, sin recalcular); con `rehacer=True` se recalculan todas. Las métricas se recalculan siempre desde las predicciones.
    """
    t0 = time.time()
    casos = casos_seleccionados(dataset, leer_veredictos())
    salida = dir_resultados(dataset)
    salida.mkdir(parents=True, exist_ok=True)
    ckpt = descargar_checkpoint()
    sha_ckpt = _sha256(ckpt)
    dev = elegir_dispositivo(dispositivo)
    predictor = None  # se construye solo si hace falta correr el modelo

    filas_corte: list[dict] = []
    excluidos: list[dict] = []
    cfg_casos: dict = {}
    n_corridas = 0
    for caso in casos:
        n, pac, nucleo = caso["n"], caso["paciente"], sorted(caso["nucleo"])
        exp = mascara_experta(n)
        imgs = None
        cfg_casos[n] = dict(paciente=pac, nucleo=nucleo, tipos={})
        for tipo in tipos:
            pistas_c = cortes_pista(nucleo, tipo)
            ev = cortes_evaluacion(nucleo, pistas_c)
            pistas = construir_pistas(exp, pistas_c, tipo)
            motivo = None
            if len(ev) < MIN_CORTES_EVAL:
                motivo = f"menos de {MIN_CORTES_EVAL} cortes de evaluación ({len(ev)})"
            elif not pistas:
                motivo = "ningún hueso con pista en los cortes de pista"
            if motivo:
                excluidos.append(dict(n=n, paciente=pac, tipo=tipo, motivo=motivo,
                                      cortes_pista=" ".join(map(str, pistas_c)), n_evaluacion=len(ev)))
                cfg_casos[n]["tipos"][tipo] = dict(cortes_pista=pistas_c, cortes_evaluacion=ev, excluido=motivo)
                continue
            ruta = _ruta_pred(dataset, n, tipo)
            huella = _huella_pistas(tipo, pistas_c, pistas)
            meta = None if rehacer else _leer_cache(ruta, huella)  # None: sin caché o inválida
            if meta is not None:
                pred, origen = meta.pop("pred"), "cache"
            else:
                if predictor is None:
                    predictor = construir_predictor(ckpt, dev)
                if imgs is None:
                    imgs = stir_preprocesada(n, dataset)
                t1 = time.time()
                pred = segmentar_volumen(predictor, imgs, pistas)
                meta = dict(dispositivo=str(dev), segundos=round(time.time() - t1, 2), checkpoint_sha256=sha_ckpt)
                ruta.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(ruta, pred=pred, dispositivo=np.array(meta["dispositivo"]),
                                    segundos=np.array(meta["segundos"]), checkpoint_sha256=np.array(sha_ckpt),
                                    huella_pistas=np.array(huella))
                origen = "modelo"
                n_corridas += 1
            objetos = sorted({o for por in pistas.values() for o in por})
            dice_pista = {c: {h: _dice(pred[c] == h, exp[c] == h) for h in _HUESOS.values()} for c in pistas_c}
            for c in ev:
                dist = min(abs(c - p) for p in pistas_c)
                for fila in metricas_corte(pred[c], exp[c]):
                    fila["eje_mediana_mm"] = fila["eje_mediana_px"] * PX_MM
                    fila["eje_p95_mm"] = fila["eje_p95_px"] * PX_MM
                    filas_corte.append(dict(n=n, paciente=pac, tipo=tipo, corte=c, distancia_a_pista=dist, **fila))
            cfg_casos[n]["tipos"][tipo] = dict(cortes_pista=pistas_c, cortes_evaluacion=ev, objetos_con_pista=objetos,
                                               dice_en_corte_de_pista=dice_pista, origen_prediccion=origen,
                                               dispositivo_prediccion=meta["dispositivo"],
                                               segundos_modelo=meta["segundos"],
                                               checkpoint_sha256_prediccion=meta["checkpoint_sha256"])
            print(f"n={n} {tipo}: {origen} ({meta['dispositivo']}, modelo {meta['segundos']}s), "
                  f"{len(ev)} cortes evaluados", flush=True)

    filas_caso = _por_caso(filas_corte)
    resumen = resumir(filas_corte, tipos)
    _escribir_csv(salida / "metricas_por_corte.csv", COLS_CORTE, filas_corte)
    _escribir_csv(salida / "metricas_por_caso.csv",
                  ["n", "paciente", "tipo", "estructura", "n_cortes", *METRICAS, "n_vacio_pred", "n_vacio_exp"], filas_caso)
    _escribir_csv(salida / "resumen.csv", COLS_RESUMEN, resumen)
    _escribir_csv(salida / "excluidos.csv", ["n", "paciente", "tipo", "motivo", "cortes_pista", "n_evaluacion"], excluidos)

    transf = repo_root() / "anotacion/hueso-experto/transformaciones.json"
    fuentes = {}
    if transf.is_file():
        fuentes = {e["n"]: e.get("fuente") for e in json.loads(transf.read_text(encoding="utf-8"))["casos"]}
    for n in cfg_casos:
        cfg_casos[n]["fuente_mascara_experta"] = fuentes.get(n, "aprobada")
    cfg = dict(
        dataset=dataset, fecha=datetime.now().isoformat(timespec="seconds"),
        dispositivo_pedido=dispositivo, dispositivo_corrida=str(dev),
        dispositivos_prediccion=sorted({t["dispositivo_prediccion"] for c in cfg_casos.values() for t in c["tipos"].values() if "excluido" not in t}),
        segundos_modelo_total=round(sum(t["segundos_modelo"] for c in cfg_casos.values() for t in c["tipos"].values() if "excluido" not in t), 1), versiones=_versiones(),
        checkpoint=dict(archivo=CHECKPOINT_FILE, sha256=sha_ckpt),
        parametros=dict(tipos=list(tipos), min_cortes_eval=MIN_CORTES_EVAL),
        n_casos=len(casos), n_corridas_con_metricas=sum(1 for c in cfg_casos.values() for t in c["tipos"].values() if "excluido" not in t),
        n_corridas_modelo_nuevas=n_corridas, n_excluidos=len(excluidos),
        tiempo_total_s=round(time.time() - t0, 1), casos=cfg_casos,
    )
    (salida / "configuracion.json").write_text(
        json.dumps(_json_ok(cfg), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return cfg
