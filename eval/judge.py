"""Rúbrica de calidad del proceso evaluada por un LLM juez.

El juez es un modelo distinto del actor y no recibe el veredicto de la
meta: puntúa solo lo que ve en la traza. La salida se valida con
`structured_call` del M2.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field

from mia_agents._env import load_env_files
from mia_agents.llm_client import BedrockProvider, LLMClient

# Nova Pro no admite on-demand con el id pelado: hay que invocarlo por el
# inference profile entre regiones.
JUDGE_MODEL_DEFAULT = "us.amazon.nova-pro-v1:0"
MAX_TRACE_CALLS = 40
MAX_OUTPUT_CHARS = 400


class VeredictoRubrica(BaseModel):
    """Puntuaciones de 1 (muy malo) a 5 (muy bueno)."""

    exploracion_sistematica: int = Field(
        ge=1,
        le=5,
        description=(
            "¿Exploró con método (mirar, examinar contenedores, deducir dónde "
            "buscar) o probó a ciegas? 5 = cada acción se apoya en lo que "
            "acababa de observar. 1 = tanteo aleatorio."
        ),
    )
    uso_de_memoria: int = Field(
        ge=1,
        le=5,
        description=(
            "¿Aprovechó lo que ya había observado? 5 = nunca repite una acción "
            "resuelta ni vuelve a mirar lo que ya sabe. 1 = repite llamadas "
            "idénticas u olvida objetos que ya tenía."
        ),
    )
    fidelidad_del_reporte: int = Field(
        ge=1,
        le=5,
        description=(
            "¿La respuesta final describe lo que realmente pasó en la traza? "
            "5 = fiel, incluido admitir que no lo logró. 1 = afirma logros que "
            "la traza contradice."
        ),
    )
    recuperacion_de_errores: int = Field(
        ge=1,
        le=5,
        description=(
            "Ante una salida que empieza con 'Error:', ¿corrigió el argumento? "
            "5 = corrige a la primera. 1 = reintenta lo mismo o lo ignora. Si "
            "no hubo errores en la traza, puntuá 5."
        ),
    )
    justificacion: str = Field(
        max_length=600,
        description=(
            "Dos o tres frases que expliquen las puntuaciones, citando pasos "
            "concretos de la traza."
        ),
    )


_PROMPT = """\
Sos un evaluador de agentes de IA. Te paso la traza de un agente que intentaba \
resolver una sala de escape en un mundo simulado, y tenés que puntuar la \
CALIDAD DE SU PROCESO con la rúbrica de la herramienta.

Evaluá el proceso, no el desenlace. No te digo si cumplió la meta y no lo \
asumas: un agente puede tropezarse con la solución jugando pésimo, y otro \
puede razonar impecablemente y quedarse a un paso. Puntuá lo que ves en la \
traza.

META DEL ESCENARIO (lo que se le pidió al agente)
{user_message}

TRAZA DE ACCIONES (en orden; cada una con sus argumentos y lo que devolvió el mundo)
{trace}

RESPUESTA FINAL QUE DIO EL AGENTE
{answer}

Invocá final_result con las cuatro puntuaciones enteras de 1 a 5 y una \
justificación breve que cite pasos concretos.\
"""


def _formatear_traza(run: dict[str, Any]) -> str:
    """Traza legible, acotada para que una corrida larga no dispare el coste."""
    lineas: list[str] = []
    llamadas = run.get("tool_calls", [])
    for c in llamadas[:MAX_TRACE_CALLS]:
        args = c.get("arguments") or "{}"
        salida = (c.get("output") or "").replace("\n", " ")
        if len(salida) > MAX_OUTPUT_CHARS:
            salida = salida[:MAX_OUTPUT_CHARS] + " […recortado]"
        marca = " [ERROR]" if c.get("is_error") else ""
        lineas.append(f"{c['index'] + 1}. {c.get('name')}({args}){marca} -> {salida}")
    if len(llamadas) > MAX_TRACE_CALLS:
        lineas.append(
            f"[…{len(llamadas) - MAX_TRACE_CALLS} llamadas más, omitidas por longitud]"
        )
    return "\n".join(lineas) or "(el agente no invocó ninguna herramienta)"


def modelo_juez() -> str:
    return os.environ.get("MIA_JUDGE_MODEL", JUDGE_MODEL_DEFAULT)


def construir_juez(model_id: str | None = None) -> Any:
    """`MyAgent` apuntando al modelo juez."""
    from student_framework.agent import MyAgent

    # `BedrockProvider` construido a mano no dispara la carga del .env que
    # sí hace `LLMClient.from_env()`.
    load_env_files()
    modelo = model_id or modelo_juez()
    return MyAgent(
        llm_client=LLMClient(BedrockProvider(model=modelo)),
        system_prompt=(
            "Sos un evaluador riguroso y conciso de agentes de IA. Puntuás con "
            "criterio: reservás el 5 para procesos realmente buenos y no "
            "inflás las notas."
        ),
        max_history_messages=20,
    )


def juzgar(run: dict[str, Any], user_message: str, juez: Any = None) -> dict[str, Any]:
    """Puntúa una corrida.

    Un fallo del juez se degrada a un dict con `error`: la métrica
    cuantitativa ya está calculada y no depende de esto.
    """
    juez = juez or construir_juez()
    prompt = _PROMPT.format(
        user_message=user_message,
        trace=_formatear_traza(run),
        answer=run.get("answer") or "(sin respuesta)",
    )
    try:
        veredicto = juez.structured_call(prompt, VeredictoRubrica)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    datos = veredicto.model_dump()
    datos["modelo_juez"] = modelo_juez()
    return datos


DIMENSIONES = [
    "exploracion_sistematica",
    "uso_de_memoria",
    "fidelidad_del_reporte",
    "recuperacion_de_errores",
]


def promedios_rubrica(veredictos: list[dict[str, Any]]) -> dict[str, float | None]:
    validos = [v for v in veredictos if v and "error" not in v]
    if not validos:
        return {d: None for d in DIMENSIONES}
    return {d: sum(v[d] for v in validos) / len(validos) for d in DIMENSIONES}
