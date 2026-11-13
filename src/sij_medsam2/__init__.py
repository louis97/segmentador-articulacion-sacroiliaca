"""Evaluación de MedSAM2 con pistas de hueso experto sobre RM STIR de la articulación sacroilíaca."""
from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__", "ejecutar_pipeline"]


def __getattr__(nombre: str):
    # Importación perezosa: `pipeline` arrastra torch y MedSAM2, que son lentos de importar.
    if nombre == "ejecutar_pipeline":
        from .pipeline import ejecutar_pipeline

        return ejecutar_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {nombre!r}")
