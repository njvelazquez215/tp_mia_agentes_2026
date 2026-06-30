"""
Contador de Palabras.

Demuestra el mismo patrón que las otras dos (`callable` tipado +
`ToolSchema.from_callable` + registro). Cuenta palabras, caracteres y
líneas de un texto.
"""

from __future__ import annotations
from typing import Annotated
from pydantic import Field
from mia_agents.types import ToolSchema


def count_words(
    text: Annotated[
        str,
        Field(description="El texto sobre el que contar palabras, caracteres y líneas."),
    ],
) -> str:
    """Cuenta palabras, caracteres y líneas del texto recibido.

    Una palabra es cualquier secuencia separada por espacios en blanco.
    Devuelve un resumen legible: número de palabras, de caracteres (con
    espacios) y de líneas.
    """
    words = len(text.split())
    characters = len(text)
    # Un texto no vacío tiene al menos una línea; el vacío tiene cero.
    lines = text.count("\n") + 1 if text else 0

    return (
        f"palabras: {words}, caracteres: {characters}, líneas: {lines}"
    )


count_words_schema = ToolSchema.from_callable(count_words)
