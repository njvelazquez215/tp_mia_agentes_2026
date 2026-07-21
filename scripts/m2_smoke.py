"""Smoke test de M2 contra un LLM real (Bedrock u Ollama).

Los tests de conformidad corren con `MockLLMClient`; esto verifica lo que
el mock no puede: que la ventana recortada, los mensajes de reparación y
los resultados de herramienta viajen bien por el provider real.

    python scripts/m2_smoke.py

Requiere el proveedor configurado en `.env` (BEDROCK_MODEL_ID o OLLAMA_HOST)
y hace llamadas reales, así que consume tokens.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel, Field

from mia_agents.llm_client import LLMClient
from mia_agents.types import LLMResponse

from student_framework import build_agent
from student_framework.agent import StructuredOutputError

PRESUPUESTO = 8


class _Espia:
    """Envuelve al cliente real para ver qué se le manda en cada llamada."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.tamanios: list[int] = []

    def chat(self, messages, tools=None, system=None, temperature=0.2,
             response_format=None) -> LLMResponse:
        self.tamanios.append(len(messages))
        roles = "".join(m.get("role", "?")[0] for m in messages)
        print(f"    -> chat con {len(messages)} mensajes [{roles}]")
        return self._inner.chat(
            messages=messages,
            tools=tools,
            system=system,
            temperature=temperature,
            response_format=response_format,
        )


class Resumen(BaseModel):
    operacion: str = Field(description="La cuenta que se resolvió, como texto.")
    resultado: int = Field(description="El resultado numérico de la cuenta.")


def titulo(texto: str) -> None:
    print(f"\n{'=' * 70}\n{texto}\n{'=' * 70}")


def main() -> int:
    espia = _Espia(LLMClient.from_env())
    agent = build_agent({"llm_client": espia, "max_history_messages": PRESUPUESTO})

    titulo(f"1. Conversación multiturno (presupuesto: {PRESUPUESTO} mensajes)")
    turnos = [
        "Guardá este dato: el código de la sala es ALFA-7. Respondé solo 'anotado'.",
        "Calculá 17 * 23 con la calculadora.",
        "Ahora calculá 391 % 7.",
        "Contá las palabras de: 'rosas son rojas violetas azules'.",
        "¿Cuál era el código de la sala que te di al principio?",
    ]
    for i, turno in enumerate(turnos, 1):
        print(f"\n[turno {i}] {turno}")
        result = agent.run(turno)
        print(f"  answer: {result.answer.strip()[:200]}")
        print(f"  steps: {[s.tool_name for s in result.steps]}")
        print(f"  tokens: in={result.input_tokens} out={result.output_tokens}")
        assert result.answer, "answer vacío"

    maximo = max(espia.tamanios)
    print(f"\n  máximo de mensajes enviados en una llamada: {maximo} (tope {PRESUPUESTO})")
    assert maximo <= PRESUPUESTO, "la ventana se pasó del presupuesto"
    print(
        "  NOTA: con este presupuesto el turno 1 ya salió de la ventana, así que\n"
        "  el modelo no puede recordar ALFA-7. Es el tradeoff de sliding window."
    )

    titulo("1b. La misma conversación con presupuesto holgado sí recuerda")
    agent_amplio = build_agent({"llm_client": espia, "max_history_messages": 40})
    agent_amplio.run("Guardá este dato: el código de la sala es ALFA-7.")
    agent_amplio.run("Calculá 17 * 23 con la calculadora.")
    recordado = agent_amplio.run("¿Cuál era el código de la sala?")
    print(f"  answer: {recordado.answer.strip()[:200]}")
    print(f"  ¿recordó ALFA-7?: {'ALFA-7' in recordado.answer}")

    titulo("2. Recuperación de la calculadora (operador inválido)")
    result = agent.run(
        "Usá la calculadora con operator='^' para 2 y 8. Si falla, resolvelo "
        "con los operadores que sí existan y decime el resultado."
    )
    print(f"  answer: {result.answer.strip()[:300]}")
    for step in result.steps:
        print(f"  step {step.tool_name}: out={str(step.tool_output)[:120]!r}")

    titulo("3. Recuperación del lector de archivos (nombre inexistente)")
    result = agent.run(
        "Leé el archivo 'requirement.txt' de este directorio. Si no existe, "
        "fijate en el listado que devuelve la herramienta y leé el correcto."
    )
    print(f"  answer: {result.answer.strip()[:300]}")
    for step in result.steps:
        print(f"  step {step.tool_name}: out={str(step.tool_output)[:120]!r}")

    titulo("4. Salida estructurada con final_result")
    try:
        parsed = agent.structured_call(
            prompt="Resolvé 17 * 23 y devolvé la operación y el resultado.",
            schema=Resumen,
        )
        print(f"  parsed: {parsed!r}")
        assert isinstance(parsed, Resumen)
    except StructuredOutputError as exc:
        print(f"  falló limpio (aceptable): {exc}")

    titulo("OK: el agente sobrevivió al proveedor real")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
