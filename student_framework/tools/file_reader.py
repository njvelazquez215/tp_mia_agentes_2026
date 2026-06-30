"""
Lector de archivos de texto (E/S restringida).

Lee un archivo de texto (UTF-8) y devuelve su contenido como cadena. Los
errores habituales (archivo inexistente, ruta que es un directorio,
contenido binario no decodificable) se devuelven como mensajes de error
legibles en lugar de propagar excepciones, para que el bucle del agente
pueda observar el fallo y seguir razonando.
"""

from __future__ import annotations
from pathlib import Path
from typing import Annotated
from pydantic import Field
from mia_agents.types import ToolSchema

# Límite para evitar poner archivos enormes al contexto del LLM.
_MAX_BYTES = 100_000


def read_text_file(
    path: Annotated[
        str,
        Field(description="Ruta al archivo de texto a leer (relativa o absoluta)."),
    ],
) -> str:
    """Lee un archivo de texto UTF-8 y devuelve su contenido como cadena.

    Solo admite archivos de texto. Si el archivo no existe, no es texto
    válido o supera el límite de tamaño, devuelve un mensaje de error
    descriptivo (no lanza excepción).
    """
    file_path = Path(path)

    if not file_path.exists():
        return f"Error: el archivo '{path}' no existe."
    if file_path.is_dir():
        return f"Error: '{path}' es un directorio, no un archivo."

    try:
        size = file_path.stat().st_size
        if size > _MAX_BYTES:
            return (
                f"Error: el archivo '{path}' es demasiado grande "
                f"({size} bytes > {_MAX_BYTES} bytes permitidos)."
            )
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return (
            f"Error: '{path}' no es un archivo de texto UTF-8 válido "
            "(¿es binario?)."
        )
    except OSError as exc:
        return f"Error al leer '{path}': {exc}"


read_text_file_schema = ToolSchema.from_callable(read_text_file)
