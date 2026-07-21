"""Consola interactiva para probar el agente a mano.

    python scripts/chat.py                  # presupuesto por defecto (50)
    python scripts/chat.py --presupuesto 6  # para ver la ventana en acción

Escribí mensajes y respondé como en cualquier chat. Comandos disponibles:

    /ayuda          lista los comandos
    /debug          muestra (o esconde) lo que se le manda al LLM en cada llamada
    /historial      imprime el historial que el agente está reteniendo
    /tokens         total de tokens consumidos en la sesión
    /reset          borra la conversación y arranca de cero
    /estructurado   pide una respuesta validada con Pydantic (structured_call)
    /salir          terminar

Requiere el proveedor configurado en `.env` y hace llamadas reales.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# La consola de Windows suele venir en cp1252 y revienta al imprimir
# acentos o el contenido que devuelve el modelo.
for flujo in (sys.stdout, sys.stderr):
    reconfigure = getattr(flujo, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")

from pydantic import BaseModel, Field

from mia_agents.llm_client import LLMClient
from mia_agents.types import LLMResponse

from student_framework import build_agent
from student_framework.agent import StructuredOutputError

SYSTEM = (
    "Sos un asistente útil. Respondé siempre en español, de forma breve y "
    "directa. No escribas bloques <thinking> ni expliques tu razonamiento "
    "interno: dame solo la respuesta final. Usá las herramientas disponibles "
    "cuando hagan falta y confiá en la conversación previa como memoria."
)


class Ficha(BaseModel):
    """Schema de ejemplo para el comando /estructurado."""

    tema: str = Field(description="De qué trata la respuesta, en pocas palabras.")
    respuesta: str = Field(description="La respuesta al pedido del usuario.")
    confianza: int = Field(description="Qué tan seguro estás, de 0 a 100.")


class ClienteVerboso:
    """Envuelve el cliente real para poder mostrar qué se le manda."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.debug = False
        self.total_in = 0
        self.total_out = 0

    def chat(self, messages, tools=None, system=None, temperature=0.2,
             response_format=None) -> LLMResponse:
        if self.debug:
            print(f"\n  --- ventana enviada al LLM: {len(messages)} mensajes ---")
            for m in messages:
                texto = str(m.get("content") or "")
                if m.get("tool_calls"):
                    nombres = [
                        tc["function"]["name"] for tc in m["tool_calls"]
                    ]
                    texto = f"(pide herramientas: {', '.join(nombres)}) {texto}"
                print(f"  |  [{m.get('role'):<9}] {_corto(texto, 90)}")
            print("  ---")

        resp = self._inner.chat(
            messages=messages, tools=tools, system=system,
            temperature=temperature, response_format=response_format,
        )
        self.total_in += resp.input_tokens or 0
        self.total_out += resp.output_tokens or 0
        return resp


def _corto(texto: str, n: int) -> str:
    texto = " ".join(texto.split())
    return texto if len(texto) <= n else texto[: n - 3] + "..."


def _mostrar_resultado(result: Any) -> None:
    print(f"\nagente> {result.answer.strip()}")

    if result.steps:
        print("\n  herramientas usadas:")
        for step in result.steps:
            # `step.error` es un fallo del agente (tool inexistente, JSON
            # roto). Un "Error:" en la salida es un error recuperable que
            # la herramienta devolvió a propósito para que el LLM corrija.
            if step.error:
                estado = "FALLO"
            elif str(step.tool_output).startswith("Error:"):
                estado = "recuperable"
            else:
                estado = "ok"
            print(f"    - {step.tool_name} [{estado}]")
            print(f"        argumentos: {_corto(str(step.tool_input), 100)}")
            salida = step.error if step.error else step.tool_output
            print(f"        resultado:  {_corto(str(salida), 100)}")

    if result.error:
        print(f"\n  aviso: {result.error}")
    print(f"\n  tokens de este turno: entrada={result.input_tokens} salida={result.output_tokens}")


def _ayuda() -> None:
    print(
        "\ncomandos:\n"
        "  /debug         muestra u oculta la ventana que se le manda al LLM\n"
        "  /historial     imprime el historial retenido por el agente\n"
        "  /tokens        total de tokens de la sesión\n"
        "  /reset         borra la conversación\n"
        "  /estructurado  <pedido>  respuesta validada con Pydantic\n"
        "  /salir         terminar\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Consola interactiva del agente.")
    parser.add_argument(
        "--presupuesto",
        type=int,
        default=50,
        help="max_history_messages: cuántos mensajes entran en la ventana.",
    )
    args = parser.parse_args()

    cliente = ClienteVerboso(LLMClient.from_env())
    agent = build_agent(
        {
            "llm_client": cliente,
            "max_history_messages": args.presupuesto,
            "system_prompt": SYSTEM,
        }
    )

    print("=" * 70)
    print(f"Agente MIA303 - presupuesto de contexto: {args.presupuesto} mensajes")
    print("Escribí tu mensaje, o /ayuda para ver los comandos.")
    print("=" * 70)

    while True:
        try:
            entrada = input("\nvos> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nchau")
            return 0

        if not entrada:
            continue

        if entrada in ("/salir", "/exit", "/quit"):
            print("chau")
            return 0

        if entrada == "/ayuda":
            _ayuda()
            continue

        if entrada == "/debug":
            cliente.debug = not cliente.debug
            print(f"  debug {'activado' if cliente.debug else 'desactivado'}")
            continue

        if entrada == "/tokens":
            print(
                f"  sesión: entrada={cliente.total_in} salida={cliente.total_out} "
                f"total={cliente.total_in + cliente.total_out}"
            )
            continue

        if entrada == "/reset":
            agent.reset()
            print("  conversación borrada")
            continue

        if entrada == "/historial":
            historial = agent.history
            print(f"\n  el agente retiene {len(historial)} mensajes:")
            for m in historial:
                print(f"    [{m.get('role'):<9}] {_corto(str(m.get('content') or ''), 90)}")
            continue

        if entrada.startswith("/estructurado"):
            pedido = entrada[len("/estructurado"):].strip()
            if not pedido:
                print("  usá: /estructurado <lo que quieras preguntar>")
                continue
            try:
                ficha = agent.structured_call(prompt=pedido, schema=Ficha)
            except StructuredOutputError as exc:
                print(f"\n  el modelo no logró producir la estructura: {exc}")
                continue
            print("\n  objeto validado (instancia de Ficha):")
            print(f"    tema:       {ficha.tema}")
            print(f"    respuesta:  {ficha.respuesta}")
            print(f"    confianza:  {ficha.confianza}")
            continue

        if entrada.startswith("/"):
            print("  comando desconocido; probá /ayuda")
            continue

        try:
            _mostrar_resultado(agent.run(entrada))
        except Exception as exc:  # noqa: BLE001 - la consola no debe morirse
            print(f"\n  la llamada falló: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
