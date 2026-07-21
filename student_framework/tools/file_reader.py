"""
Lector de archivos de texto (E/S restringida).

Lee un archivo de texto (UTF-8) dentro de un sandbox y devuelve su
contenido como cadena. El sandbox es el directorio de trabajo actual,
salvo que se defina la variable de entorno `MIA_FILE_SANDBOX`.

M2: los errores de uso no lanzan excepción y son *accionables*. Además de
decir qué falló, la herramienta le da al LLM el dato que necesita para
corregir el intento: la regla de rutas que violó, o el listado del
directorio donde buscaba.
"""

from __future__ import annotations
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Annotated
from pydantic import Field
from mia_agents.types import ToolSchema

# Límite para evitar poner archivos enormes al contexto del LLM.
_MAX_BYTES = 100_000
# Cuántas entradas listamos como pista antes de cortar.
_MAX_LISTADO = 40

_REGLA_RUTA = (
    "Una ruta válida es relativa al directorio de trabajo, sin '..' y sin "
    "unidad ni barra inicial (por ejemplo: 'datos/notas.txt')."
)


class _RutaInvalida(ValueError):
    """La ruta viola las reglas del sandbox."""


def _sandbox_root() -> Path:
    """Raíz permitida: `MIA_FILE_SANDBOX` si está definida, si no el cwd."""
    configurada = os.environ.get("MIA_FILE_SANDBOX")
    base = Path(configurada) if configurada else Path.cwd()
    return base.resolve()


def _es_absoluta(raw: str) -> bool:
    """Detecta rutas absolutas de ambos estilos, corra donde corra el test."""
    return (
        raw.startswith(("/", "\\"))
        or PureWindowsPath(raw).is_absolute()
        or PurePosixPath(raw).is_absolute()
    )


def _resolver(raw: str, root: Path) -> Path:
    """Valida la ruta contra las reglas del sandbox y la resuelve."""
    if not raw or not raw.strip():
        raise _RutaInvalida(f"la ruta está vacía. {_REGLA_RUTA}")

    candidata = raw.strip()
    if _es_absoluta(candidata):
        raise _RutaInvalida(
            f"'{raw}' es una ruta absoluta y solo se admiten rutas relativas "
            f"al sandbox ('{root}'). {_REGLA_RUTA}"
        )
    if ".." in PurePosixPath(candidata.replace("\\", "/")).parts:
        raise _RutaInvalida(
            f"'{raw}' usa '..' para subir de directorio, y eso no está "
            f"permitido. {_REGLA_RUTA}"
        )

    resuelta = (root / candidata).resolve()
    if resuelta != root and root not in resuelta.parents:
        raise _RutaInvalida(
            f"'{raw}' apunta fuera del sandbox ('{root}'). {_REGLA_RUTA}"
        )
    return resuelta


def _listar(directorio: Path, root: Path) -> str:
    """Listado corto y relativo al sandbox, para orientar al LLM."""
    try:
        entradas = sorted(directorio.iterdir(), key=lambda p: p.name)
    except OSError:
        return ""
    # Las entradas ocultas (.env, .git/, .venv/) no son lo que el modelo
    # está buscando y solo agregan ruido al contexto.
    entradas = [e for e in entradas if not e.name.startswith(".")]
    if not entradas:
        return "(el directorio está vacío)"

    nombres = [
        f"{e.name}/" if e.is_dir() else e.name for e in entradas[:_MAX_LISTADO]
    ]
    restantes = len(entradas) - len(nombres)
    if restantes > 0:
        nombres.append(f"... (+{restantes} más)")
    return ", ".join(nombres)


def read_text_file(
    path: Annotated[
        str,
        Field(
            description=(
                "Ruta relativa al directorio de trabajo del archivo de texto a "
                "leer. No se admiten rutas absolutas ni '..'."
            )
        ),
    ],
) -> str:
    """Lee un archivo de texto UTF-8 y devuelve su contenido como cadena.

    Solo admite archivos de texto ubicados dentro del directorio de
    trabajo (sandbox). Si la ruta es inválida, el archivo no existe, es un
    directorio, no es texto válido o supera el límite de tamaño, devuelve
    un mensaje de error que explica el problema y cómo corregirlo (no
    lanza excepción).
    """
    root = _sandbox_root()

    try:
        file_path = _resolver(path, root)
    except _RutaInvalida as exc:
        return f"Error: {exc}"

    if not file_path.exists():
        contenedor = file_path.parent
        if contenedor.is_dir():
            disponibles = _listar(contenedor, root)
            relativo = contenedor.relative_to(root).as_posix() or "."
            return (
                f"Error: el archivo '{path}' no existe. En '{relativo}' hay: "
                f"{disponibles}. Elegí uno de esos nombres y reintentá."
            )
        return (
            f"Error: el archivo '{path}' no existe y su directorio contenedor "
            "tampoco. Verificá la ruta relativa antes de reintentar."
        )

    if file_path.is_dir():
        return (
            f"Error: '{path}' es un directorio, no un archivo. Contiene: "
            f"{_listar(file_path, root)}. Reintentá con la ruta de uno de "
            "esos archivos."
        )

    try:
        size = file_path.stat().st_size
        if size > _MAX_BYTES:
            return (
                f"Error: el archivo '{path}' es demasiado grande "
                f"({size} bytes > {_MAX_BYTES} bytes permitidos). No hay forma "
                "de leerlo entero con esta herramienta."
            )
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return (
            f"Error: '{path}' no es un archivo de texto UTF-8 válido "
            "(¿es binario?). Esta herramienta solo lee texto."
        )
    except OSError as exc:
        return f"Error al leer '{path}': {exc}"


read_text_file_schema = ToolSchema.from_callable(read_text_file)
