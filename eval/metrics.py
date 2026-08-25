"""Métricas agregadas y clasificación de modos de fallo.

Cada corrida se clasifica en dos ejes independientes: `terminacion` (por
qué se detuvo el agente) y `friccion` (qué error predominó en su traza).
Ambos se derivan de la traza, sin llamar a ningún modelo.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

from eval.configs import DIFFICULTY_ORDER

TERMINACION = {
    "exito": "El mundo cumple la meta al terminar la corrida.",
    "presupuesto_agotado": (
        "Se acabaron las iteraciones sin cumplir la meta: el agente seguía "
        "actuando cuando lo cortamos."
    ),
    "rendicion_temprana": (
        "Devolvió una respuesta final sin cumplir la meta, teniendo "
        "presupuesto de sobra."
    ),
    "sin_accion": "No llegó a invocar ninguna herramienta.",
    "crash": "Una excepción escapó del agente y abortó la corrida.",
}

FRICCION = {
    "ninguna": "La traza no acumuló errores de herramienta relevantes.",
    "id_inexistente": (
        "Ids que no existen: usó el nombre en prosa ('llave dorada') en vez "
        "del id ('llave_oro'), o inventó un objeto."
    ),
    "objeto_no_visible": (
        "Actuó sobre objetos que existen pero no son visibles desde donde "
        "está: no examinó el contenedor que los oculta, o está en otra sala."
    ),
    "inventario_vacio": "Intentó `use` sin haber hecho `take` antes.",
    "pieza_incorrecta": (
        "Aplicó el objeto equivocado sobre una cerradura: error de "
        "razonamiento sobre qué llave abre qué."
    ),
    "navegacion_invalida": (
        "Salió por direcciones inexistentes o bloqueadas: no recuerda el mapa "
        "o no leyó las salidas."
    ),
    "argumentos_invalidos": (
        "El framework rechazó la llamada antes de llegar al mundo: "
        "herramienta inexistente, JSON malformado o firma incompatible."
    ),
    "repeticion": (
        "Repitió la misma llamada con los mismos argumentos tres veces o más."
    ),
}

# Se evalúan en orden, de más específico a más genérico, contra el texto
# que devuelven las herramientas de `mia_world`.
_PATRONES: list[tuple[str, re.Pattern[str]]] = [
    ("id_inexistente", re.compile(r"no existe ning[uú]n objeto con id", re.I)),
    ("inventario_vacio", re.compile(r"no llevas ning[uú]n", re.I)),
    (
        "navegacion_invalida",
        re.compile(r"no hay salida|est[aá] bloqueado|no hay salidas", re.I),
    ),
    ("objeto_no_visible", re.compile(r"no ves ning[uú]n|no es visible o accesible", re.I)),
    ("pieza_incorrecta", re.compile(r"pero no encaja", re.I)),
]

# Heurística para detectar respuestas finales que contradicen el estado del
# mundo. El juez LLM evalúa lo mismo con más criterio.
_CLAIM_EXITO = re.compile(
    r"\b(abr[ií]|abierta|logr[eé]|consegu[ií]|escap[eé]|sal[ií]|"
    r"he\s+abierto|est[aá]\s+abierta|misi[oó]n\s+cumplida)\b",
    re.I,
)


def clasificar_terminacion(run: dict[str, Any]) -> str:
    if run.get("crashed"):
        return "crash"
    if run["goal_achieved"]:
        return "exito"
    if run["n_calls"] == 0:
        return "sin_accion"
    if run.get("exhausted_steps"):
        return "presupuesto_agotado"
    return "rendicion_temprana"


def clasificar_friccion(run: dict[str, Any]) -> str:
    """Categoría de error más frecuente en la traza.

    Un bucle sin errores no lo detecta el conteo de errores, así que
    `repeticion` gana cuando hay llamadas idénticas repetidas y no
    predomina ninguna otra categoría.
    """
    conteo: Counter[str] = Counter()
    for call in run.get("tool_calls", []):
        if not call.get("is_error"):
            continue
        if call.get("framework_error"):
            conteo["argumentos_invalidos"] += 1
            continue
        salida = call.get("output") or ""
        for etiqueta, patron in _PATRONES:
            if patron.search(salida):
                conteo[etiqueta] += 1
                break
        else:
            conteo["argumentos_invalidos"] += 1

    firmas = [(c.get("name"), c.get("arguments")) for c in run.get("tool_calls", [])]
    repetidas = Counter(firmas)
    hay_bucle = any(n >= 3 for n in repetidas.values())

    if not conteo:
        return "repeticion" if hay_bucle else "ninguna"
    etiqueta, veces = conteo.most_common(1)[0]
    if hay_bucle and veces < max(repetidas.values()):
        return "repeticion"
    return etiqueta


def afirma_exito_falso(run: dict[str, Any]) -> bool:
    """El agente dice haber cumplido la meta pero el mundo dice que no."""
    if run["goal_achieved"]:
        return False
    return bool(_CLAIM_EXITO.search(run.get("answer") or ""))


@dataclass
class Agregado:
    etiqueta: str
    n_runs: int
    n_exitos: int
    goal_rate: float
    goal_rate_std: float
    pass_at_k: float
    k: int
    eficiencia_media: float | None
    calls_media: float
    tasa_error_calls: float
    tokens_in: int
    tokens_out: int
    coste_usd: float
    segundos: float
    afirmaciones_falsas: int
    terminacion: dict[str, int]
    friccion: dict[str, int]


def _pass_at_k(runs: list[dict[str, Any]]) -> tuple[float, int]:
    """Escenarios resueltos en al menos una repetición.

    La brecha contra el goal rate mide varianza: si pass@k es mucho mayor,
    el agente puede resolverlo pero no de forma fiable.
    """
    por_escenario: dict[str, list[bool]] = {}
    for r in runs:
        por_escenario.setdefault(r["scenario_id"], []).append(r["goal_achieved"])
    if not por_escenario:
        return 0.0, 0
    k = max(len(v) for v in por_escenario.values())
    resueltos = sum(1 for v in por_escenario.values() if any(v))
    return resueltos / len(por_escenario), k


def agregar(runs: list[dict[str, Any]], etiqueta: str) -> Agregado:
    if not runs:
        return Agregado(
            etiqueta, 0, 0, 0.0, 0.0, 0.0, 0, None, 0.0, 0.0, 0, 0, 0.0, 0.0, 0, {}, {}
        )

    exitos = [r["goal_achieved"] for r in runs]
    binarios = [1.0 if e else 0.0 for e in exitos]
    eficiencias = [r["efficiency"] for r in runs if r.get("efficiency") is not None]
    total_calls = sum(r["n_calls"] for r in runs)
    total_errores = sum(r["n_error_calls"] for r in runs)
    pak, k = _pass_at_k(runs)

    return Agregado(
        etiqueta=etiqueta,
        n_runs=len(runs),
        n_exitos=sum(exitos),
        goal_rate=mean(binarios),
        goal_rate_std=pstdev(binarios) if len(binarios) > 1 else 0.0,
        pass_at_k=pak,
        k=k,
        eficiencia_media=mean(eficiencias) if eficiencias else None,
        calls_media=mean(r["n_calls"] for r in runs),
        tasa_error_calls=(total_errores / total_calls) if total_calls else 0.0,
        tokens_in=sum(r.get("input_tokens") or 0 for r in runs),
        tokens_out=sum(r.get("output_tokens") or 0 for r in runs),
        coste_usd=sum(r.get("cost_usd") or 0.0 for r in runs),
        segundos=sum(r.get("wall_seconds") or 0.0 for r in runs),
        afirmaciones_falsas=sum(1 for r in runs if afirma_exito_falso(r)),
        terminacion=dict(Counter(clasificar_terminacion(r) for r in runs)),
        friccion=dict(
            Counter(clasificar_friccion(r) for r in runs if not r["goal_achieved"])
        ),
    )


def agrupar(runs: list[dict[str, Any]], clave: str) -> dict[str, list[dict[str, Any]]]:
    grupos: dict[str, list[dict[str, Any]]] = {}
    for r in runs:
        grupos.setdefault(str(r[clave]), []).append(r)
    return grupos


def orden_dificultad(runs: Iterable[dict[str, Any]]) -> list[str]:
    vistos = {r["difficulty"] for r in runs}
    return sorted(vistos, key=lambda d: DIFFICULTY_ORDER.get(d, 99))


def escribir_jsonl(path: Path, registros: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in registros:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def leer_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
