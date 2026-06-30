"""Implementación de su agente.

Completen `register_tool` y `run` para el Milestone 1.
En el Milestone 2 amplíen `MyAgent` para que sea estatal y respete
`max_history_messages`.

Los tests de conformidad en `tests/conformance/test_m1.py` y
`test_m2.py` describen con precisión qué comportamientos deben funcionar
— léanlos antes de empezar.
"""

from __future__ import annotations
import json
from typing import Any, Callable
from mia_agents.protocols import LLMClient
from mia_agents.types import AgentResult, AgentStep, ToolCall, ToolSchema


class MyAgent:
    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt: str = "Eres un asistente útil.",
        max_iterations: int = 10,
        max_history_messages: int = 50,
    ) -> None:
        """Inicializa el agente.

        Parameters
        ----------
        llm_client : LLMClient
            Cliente LLM (real o mock) que el agente utilizará.
        system_prompt : str
            System prompt por defecto.
        max_iterations : int
            Tope de iteraciones del bucle del agente (M1).
        max_history_messages : int
            Número máximo de mensajes que se permiten en la lista
            `messages` enviada al LLM en una única llamada. En M1 este
            valor es ignorado; el agente sólo necesita aceptarlo en su
            constructor. En M2 deben respetarlo: la longitud de la
            lista de mensajes pasada a `self._llm.chat(...)` no puede
            superar este número en ninguna llamada, sin importar la
            estrategia de memoria que elijan.
        """
        self._llm = llm_client
        self._system = system_prompt
        self._max_iterations = max_iterations
        self._max_history_messages = max_history_messages
        # Estado de las herramientas registradas. Dos diccionarios
        # indexados por nombre de esquema: uno guarda el callable a
        # ejecutar, el otro el ToolSchema a exponer al LLM en `chat`.
        self._tools: dict[str, Callable[..., str]] = {}
        self._schemas: dict[str, ToolSchema] = {}

    def register_tool(
        self,
        tool: Callable[..., str],
        schema: ToolSchema,
    ) -> None:
        """Registra una herramienta callable junto a su esquema.

        El esquema suele obtenerse con `ToolSchema.from_callable(fn)`. En
        `run`, pasá `tools=list(self._schemas.values())`; el cliente LLM
        aplica `to_llm_spec()` al llamar al proveedor.

        El callable se invoca con kwargs que coinciden con la firma.
        Debe devolver una cadena.
        """
        self._tools[schema.name] = tool
        self._schemas[schema.name] = schema

    def run(self, user_message: str) -> AgentResult:
        """Ejecuta el bucle del agente hasta una respuesta final o hasta max_iterations.

        Comportamiento esperado (consulta tests/conformance/test_m1.py
        para el contrato exacto del M1):
          - Llama a `self._llm.chat(..., tools=list(self._schemas.values()))`.
          - Si la respuesta contiene tool_calls, ejecuta cada uno y vuelca
            los resultados en la siguiente llamada al chat.
          - Si la respuesta solo contiene texto (sin `tool_calls`),
            devuélvelo en `AgentResult.answer`. En M1 no uses la tool
            sintética `final_result`; ese patrón es de M2 (ver README y
            ENUNCIADO_M2.md).
          - Limita el bucle a `self._max_iterations` y termina de forma
            limpia cuando se alcance.
          - Registra cada invocación de herramienta como un `AgentStep`
            dentro de `result.steps`.

        En el M2, además, llamadas sucesivas sobre la misma instancia
        deben continuar la conversación, y la longitud de la lista de
        mensajes enviada al LLM no debe superar `self._max_history_messages`.
        Acumula los tokens de entrada/salida reportados por los
        `LLMResponse` y exponlos en `AgentResult.input_tokens` /
        `AgentResult.output_tokens`.
        """
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_message}
        ]
        steps: list[AgentStep] = []
        # Tokens: permanecen None salvo que algún LLMResponse reporte
        # tokens; en ese caso se acumulan tratando None por respuesta
        # como 0 (ver docstring de AgentResult).
        input_tokens: int | None = None
        output_tokens: int | None = None

        last_content: str = ""

        for _ in range(self._max_iterations):
            resp = self._llm.chat(
                messages=messages,
                tools=list(self._schemas.values()) if self._schemas else None,
                system=self._system,
            )

            input_tokens = _accumulate(input_tokens, resp.input_tokens)
            output_tokens = _accumulate(output_tokens, resp.output_tokens)
            last_content = resp.content or ""

            # Condición de parada M1: texto sin tool_calls -> respuesta final.
            if not resp.tool_calls:
                return AgentResult(
                    answer=last_content,
                    steps=steps,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

            # Hay tool_calls: registramos el turno del asistente (con las
            # llamadas) y ejecutamos cada herramienta, volcando su salida
            # como mensajes `role: "tool"` antes de la siguiente llamada.
            messages.append(_assistant_message(resp.content, resp.tool_calls))

            for call in resp.tool_calls:
                step, tool_content = self._execute_tool_call(call)
                steps.append(step)
                messages.append(
                    {
                        "role": "tool",
                        "content": tool_content,
                        "tool_call_id": call.id,
                    }
                )

        # Se agotó el presupuesto de iteraciones sin respuesta final de
        # texto. Devolvemos igualmente un AgentResult válido.
        return AgentResult(
            answer=last_content,
            steps=steps,
            error=f"Se alcanzó el límite de {self._max_iterations} iteraciones.",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _execute_tool_call(self, call: ToolCall) -> tuple[AgentStep, str]:
        """Ejecuta un tool_call y devuelve (AgentStep, contenido para el LLM).

        Maneja con robustez tres situaciones de error sin romper el bucle:
        argumentos JSON malformados, herramienta inexistente (alucinada por
        el LLM) y excepciones lanzadas por el callable. En todos esos casos
        el `AgentStep` queda con `error` no nulo.
        """
        # Parsear los argumentos (JSON) emitidos por el LLM.
        try:
            kwargs = json.loads(call.arguments) if call.arguments else {}
            if not isinstance(kwargs, dict):
                raise ValueError("los argumentos no son un objeto JSON")
        except (json.JSONDecodeError, ValueError) as exc:
            error = f"Argumentos inválidos para '{call.name}': {exc}"
            return (
                AgentStep(
                    tool_name=call.name,
                    tool_input=call.arguments,
                    tool_output=None,
                    error=error,
                ),
                error,
            )

        # Resolver la herramienta. Si el LLM alucina un nombre que no
        # existe, no rompemos: lo registramos como paso con error.
        tool = self._tools.get(call.name)
        if tool is None:
            error = f"Herramienta desconocida: '{call.name}'."
            return (
                AgentStep(
                    tool_name=call.name,
                    tool_input=call.arguments,
                    tool_output=None,
                    error=error,
                ),
                error,
            )

        # Ejecutar el callable con los kwargs parseados.
        try:
            output = tool(**kwargs)
        except Exception as exc:  # noqa: BLE001 - el agente no debe romperse
            error = f"Error ejecutando '{call.name}': {exc}"
            return (
                AgentStep(
                    tool_name=call.name,
                    tool_input=call.arguments,
                    tool_output=None,
                    error=error,
                ),
                error,
            )

        output_str = output if isinstance(output, str) else str(output)
        return (
            AgentStep(
                tool_name=call.name,
                tool_input=call.arguments,
                tool_output=output_str,
                error=None,
            ),
            output_str,
        )

    def structured_call(
        self,
        prompt: str,
        schema: Any,
        max_repair_attempts: int = 2,
    ) -> Any:
        """Pide al LLM una respuesta validada contra `schema` (M2).

        Obligatorio: herramienta sintética `final_result` (ver
        `mia_agents.final_result_tool_schema` / `FINAL_RESULT_TOOL_NAME`).
        El agente ofrece esa tool al LLM, valida los `arguments` del
        `tool_call` y reintenta con contexto de reparación si el modelo
        responde con texto libre o con argumentos inválidos.

        Implementa esto en el M2:
          - Pasa `tools=[final_result_tool_schema(schema)]` en cada
            llamada a `chat` dentro de este método.
          - Termina solo cuando llega un `tool_call` a `final_result`
            cuyos argumentos validan con `schema.model_validate(...)`.
          - Reintenta hasta `max_repair_attempts` incluyendo el fallo en
            los mensajes (respuesta previa, mensaje `tool`, o user de
            reparación).
          - Si tras los reintentos sigue fallando, levanta una excepción
            limpia (no devuelvas valores parciales ni `None` sin avisar).

        El M1 deja esto como stub; los tests de M2 verifican el contrato.
        """
        raise NotImplementedError("M2: implementa salida estructurada con reparación")


def _accumulate(total: int | None, value: int | None) -> int | None:
    """Suma tokens manteniendo None hasta que llega el primer valor real.

    Devuelve None si tanto el acumulado como el nuevo valor son None; en
    cuanto algún `LLMResponse` reporta tokens, empieza a sumar tratando los
    None posteriores como 0 (contrato de `AgentResult.input/output_tokens`).
    """
    if value is None:
        return total
    return (total or 0) + value


def _assistant_message(
    content: str | None, tool_calls: list[ToolCall]
) -> dict[str, Any]:
    """Construye el mensaje `assistant` con sus tool_calls para el historial.

    El formato (clave `function` con `name`/`arguments`) es el que esperan
    los proveedores reales (`OllamaProvider`/`BedrockProvider`) al re-enviar
    el historial; el `MockLLMClient` lo ignora y solo registra la llamada.
    """
    return {
        "role": "assistant",
        "content": content or "",
        "tool_calls": [
            {
                "id": call.id,
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in tool_calls
        ],
    }
