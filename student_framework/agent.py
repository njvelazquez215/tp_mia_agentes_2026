"""Implementación de su agente.

M1: registro de herramientas y bucle agente (llamar al LLM, ejecutar
tool_calls, cerrar con texto sin tool_calls).

M2 agrega, siempre dentro del agente y nunca en el cliente LLM:
  - estado conversacional entre llamadas sucesivas a `run`,
  - ventana deslizante sobre el historial acotada por `max_history_messages`,
  - `structured_call` con la tool sintética `final_result` y reparación,
  - reintentos ante fallos transitorios del LLM y de las herramientas,
  - acumulación de tokens reportados por el proveedor.
"""

from __future__ import annotations
import json
import time
from typing import Any, Callable

from pydantic import ValidationError

from mia_agents.protocols import LLMClient
from mia_agents.tool_schema import FINAL_RESULT_TOOL_NAME, final_result_tool_schema
from mia_agents.types import AgentResult, AgentStep, ToolCall, ToolSchema


class StructuredOutputError(RuntimeError):
    """`structured_call` agotó los reintentos sin una salida válida."""


# Respuesta de último recurso: el contrato de M2 pide que `run` siempre
# devuelva un `answer` no vacío, incluso si el modelo no dijo nada.
_EMPTY_ANSWER = "No obtuve una respuesta final del modelo en este turno."

# Marcas de fallo transitorio (timeouts, 5xx, throttling, red). Se buscan
# tanto en el nombre de la excepción como en su mensaje porque cada SDK
# usa su propia jerarquía: botocore lanza `ThrottlingException`, httpx
# `ReadTimeout`, ollama `ResponseError`, etc.
_TRANSIENT_MARKERS = (
    "timeout",
    "timed out",
    "throttl",
    "rate limit",
    "ratelimit",
    "too many requests",
    "429",
    "500",
    "502",
    "503",
    "504",
    "service unavailable",
    "serviceunavailable",
    "internalserver",
    "temporarily unavailable",
    "connection reset",
    "connection aborted",
    "connectionerror",
    "econnreset",
    "unavailable",
)


class MyAgent:
    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt: str = "Eres un asistente útil.",
        max_iterations: int = 10,
        max_history_messages: int = 50,
        max_retries: int = 2,
        retry_backoff: float = 0.25,
    ) -> None:
        """Inicializa el agente.

        Parameters
        ----------
        llm_client : LLMClient
            Cliente LLM (real o mock) que el agente utilizará.
        system_prompt : str
            System prompt por defecto. Viaja en el parámetro `system` de
            `chat`, así que no ocupa lugar en el presupuesto de mensajes.
        max_iterations : int
            Tope de iteraciones del bucle del agente.
        max_history_messages : int
            Número máximo de mensajes que se permiten en la lista
            `messages` enviada al LLM en una única llamada a `chat`. El
            historial completo puede ser más largo; lo que se recorta es
            lo que se envía (ver `_window`).
        max_retries : int
            Reintentos ante fallos transitorios (por llamada al LLM o por
            ejecución de herramienta). Los errores no transitorios se
            propagan sin envolver.
        retry_backoff : float
            Segundos base para el backoff lineal entre reintentos.
        """
        self._llm = llm_client
        self._system = system_prompt
        self._max_iterations = max_iterations
        self._max_history_messages = max(1, max_history_messages)
        self._max_retries = max(0, max_retries)
        self._retry_backoff = retry_backoff
        # Estado de las herramientas registradas. Dos diccionarios
        # indexados por nombre de esquema: uno guarda el callable a
        # ejecutar, el otro el ToolSchema a exponer al LLM en `chat`.
        self._tools: dict[str, Callable[..., str]] = {}
        self._schemas: dict[str, ToolSchema] = {}
        # Estado conversacional (M2): se conserva entre llamadas a `run`.
        self._history: list[dict[str, Any]] = []

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

    def reset(self) -> None:
        """Descarta el estado conversacional acumulado."""
        self._history = []

    @property
    def history(self) -> list[dict[str, Any]]:
        """Copia del historial retenido (para inspección en tests)."""
        return list(self._history)

    def run(self, user_message: str) -> AgentResult:
        """Ejecuta el bucle del agente hasta una respuesta final o hasta max_iterations.

        El turno arranca desde el historial acumulado en llamadas
        anteriores, de modo que la conversación continúa. En cada llamada
        a `chat` se envía únicamente la ventana devuelta por `_window`,
        que nunca supera `max_history_messages` mensajes y siempre
        contiene el último mensaje del usuario.

        Cierra igual que en M1: texto sin `tool_calls` es la respuesta
        final. Si se agotan las iteraciones, devuelve igualmente un
        `AgentResult` válido con `error` no nulo.
        """
        self._history.append({"role": "user", "content": user_message})
        # Trabajamos sobre una copia: el turno solo se consolida al final,
        # así un fallo a mitad de camino no deja el historial mutilado.
        working: list[dict[str, Any]] = list(self._history)

        steps: list[AgentStep] = []
        # Tokens: permanecen None salvo que algún LLMResponse reporte
        # tokens; en ese caso se acumulan tratando None por respuesta
        # como 0 (ver docstring de AgentResult).
        input_tokens: int | None = None
        output_tokens: int | None = None

        answer = ""
        error: str | None = None
        finished = False

        for _ in range(self._max_iterations):
            resp = self._chat(
                self._window(working),
                tools=list(self._schemas.values()) if self._schemas else None,
            )

            input_tokens = _accumulate(input_tokens, resp.input_tokens)
            output_tokens = _accumulate(output_tokens, resp.output_tokens)

            # Condición de parada: texto sin tool_calls -> respuesta final.
            if not resp.tool_calls:
                answer = resp.content or ""
                working.append({"role": "assistant", "content": answer})
                finished = True
                break

            # Hay tool_calls: registramos el turno del asistente (con las
            # llamadas) y ejecutamos cada herramienta, volcando su salida
            # como mensajes `role: "tool"` antes de la siguiente llamada.
            answer = resp.content or answer
            working.append(_assistant_message(resp.content, resp.tool_calls))

            for call in resp.tool_calls:
                step, tool_content = self._execute_tool_call(call)
                steps.append(step)
                working.append(
                    {
                        "role": "tool",
                        "content": tool_content,
                        "tool_call_id": call.id,
                    }
                )

        if not finished:
            error = f"Se alcanzó el límite de {self._max_iterations} iteraciones."

        self._history = _prune(working, self._max_history_messages)

        return AgentResult(
            answer=answer or _EMPTY_ANSWER,
            steps=steps,
            error=error,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    # --------------------------------------------------------------------------
    # Memoria
    # --------------------------------------------------------------------------

    def _window(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Ventana deslizante sobre `messages`, acotada por el presupuesto.

        Estrategia: conservar la cola (lo más reciente es lo más útil para
        continuar el turno en curso) y anclar el último mensaje del
        usuario aunque haya quedado fuera de esa cola. Ese anclaje es la
        invariante de recencia del enunciado: se puede tirar contexto
        viejo, nunca lo que el usuario acaba de pedir.

        Lo que sale de acá tiene que ser una conversación bien formada
        para el proveedor, no solo una lista corta (ver `_sanitize`).
        """
        budget = self._max_history_messages
        cola = messages[-budget:] if len(messages) > budget else list(messages)
        anchor = _last_user_index(messages)

        if anchor is None:
            return _drop_orphan_tool_results(cola)

        pedido = messages[anchor]
        if any(m is pedido for m in cola):
            # El pedido del usuario ya entra en la cola. Solo queda
            # descartar lo que haya quedado colgando antes del primer
            # `user`, que nunca es el pedido ni nada posterior a él.
            return _drop_from_first_user(_drop_orphan_tool_results(cola))

        # El pedido quedó fuera de la cola: lo anclamos al frente y
        # cedemos un lugar de la cola para no pasarnos del tope.
        return [pedido] + _drop_orphan_tool_results(cola[1:])

    # --------------------------------------------------------------------------
    # Llamadas al LLM y a herramientas
    # --------------------------------------------------------------------------

    def _chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSchema] | None,
    ):
        """Llama al cliente LLM reintentando solo los fallos transitorios."""
        return self._with_retries(
            lambda: self._llm.chat(
                messages=messages,
                tools=tools,
                system=self._system,
            )
        )

    def _with_retries(self, action: Callable[[], Any]) -> Any:
        """Ejecuta `action` reintentando timeouts/5xx/rate limits.

        Cualquier error que no sea transitorio se propaga tal cual: no lo
        envolvemos ni lo silenciamos para que el problema real (una tool
        mal registrada, credenciales inválidas) aflore limpio.
        """
        attempt = 0
        while True:
            try:
                return action()
            except Exception as exc:  # noqa: BLE001 - se reclasifica abajo
                attempt += 1
                if attempt > self._max_retries or not _is_transient(exc):
                    raise
                if self._retry_backoff:
                    time.sleep(self._retry_backoff * attempt)

    def _execute_tool_call(self, call: ToolCall) -> tuple[AgentStep, str]:
        """Ejecuta un tool_call y devuelve (AgentStep, contenido para el LLM).

        Maneja con robustez tres situaciones de error sin romper el bucle:
        argumentos JSON malformados, herramienta inexistente (alucinada por
        el LLM) y excepciones lanzadas por el callable. En todos esos casos
        el `AgentStep` queda con `error` no nulo y el mensaje que ve el LLM
        explica cómo corregir el intento.
        """
        # Parsear los argumentos (JSON) emitidos por el LLM.
        try:
            kwargs = json.loads(call.arguments) if call.arguments else {}
            if not isinstance(kwargs, dict):
                raise ValueError("los argumentos no son un objeto JSON")
        except (json.JSONDecodeError, ValueError) as exc:
            return _failed_step(
                call,
                f"Argumentos inválidos para '{call.name}': {exc}. "
                "Reintentá enviando un objeto JSON con los parámetros del esquema.",
            )

        # Resolver la herramienta. Si el LLM alucina un nombre que no existe, no rompemos: 
        # lo registramos como paso con error y le decimos qué herramientas sí existen.
        tool = self._tools.get(call.name)
        if tool is None:
            disponibles = ", ".join(sorted(self._tools)) or "(ninguna)"
            return _failed_step(
                call,
                f"Herramienta desconocida: '{call.name}'. "
                f"Herramientas disponibles: {disponibles}.",
            )

        # Ejecutar el callable. Los fallos transitorios (una tool que sale
        # a la red) se reintentan; el resto queda como paso con error.
        try:
            output = self._with_retries(lambda: tool(**kwargs))
        except TypeError as exc:
            esperados = ", ".join(
                self._schemas[call.name].parameters.get("properties", {})
            )
            return _failed_step(
                call,
                f"Argumentos incompatibles con '{call.name}': {exc}. "
                f"Parámetros esperados: {esperados or '(ninguno)'}.",
            )
        except Exception as exc:  # noqa: BLE001 - el agente no debe romperse
            return _failed_step(call, f"Error ejecutando '{call.name}': {exc}")

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

    # --------------------------------------------------------------------------
    # Salida estructurada
    # --------------------------------------------------------------------------

    def structured_call(
        self,
        prompt: str,
        schema: Any,
        max_repair_attempts: int = 2,
    ) -> Any:
        """Pide al LLM una respuesta validada contra `schema`.

        El agente ofrece la tool sintética `final_result` (derivada del
        propio `schema`) como única herramienta y solo acepta cerrar con
        un `tool_call` a esa tool cuyos argumentos validen. Si el modelo
        responde texto libre, emite JSON malformado o argumentos que no
        pasan la validación, se reintenta hasta `max_repair_attempts`
        veces agregando al historial el intento fallido y un mensaje de
        reparación que describe exactamente qué estuvo mal.

        Se hacen como mucho `1 + max_repair_attempts` llamadas al LLM.
        Agotadas, levanta `StructuredOutputError` con el último fallo: no
        se devuelven instancias parciales ni `None`.
        """
        tool = final_result_tool_schema(schema)
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": prompt},
        ]
        last_error = ""

        for attempt in range(max_repair_attempts + 1):
            resp = self._with_retries(
                lambda: self._llm.chat(
                    messages=self._window(messages),
                    tools=[tool],
                    system=self._structured_system_prompt(),
                )
            )

            call = next(
                (c for c in resp.tool_calls if c.name == FINAL_RESULT_TOOL_NAME),
                None,
            )

            if call is None:
                last_error = (
                    "no invocaste la herramienta "
                    f"'{FINAL_RESULT_TOOL_NAME}'; respondiste texto libre"
                )
                messages.append(
                    {"role": "assistant", "content": resp.content or ""}
                )
                messages.append({"role": "user", "content": _repair_prompt(last_error)})
                continue

            try:
                parsed = _validate_final_result(call, schema)
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = str(exc)
                messages.append(_assistant_message(resp.content, [call]))
                messages.append(
                    {
                        "role": "tool",
                        "content": f"Validación fallida: {last_error}",
                        "tool_call_id": call.id,
                    }
                )
                messages.append({"role": "user", "content": _repair_prompt(last_error)})
                continue

            return parsed

        raise StructuredOutputError(
            f"El modelo no produjo una salida válida para {getattr(schema, '__name__', schema)!r} "
            f"tras {max_repair_attempts + 1} intentos. Último fallo: {last_error}"
        )

    def _structured_system_prompt(self) -> str:
        return (
            f"{self._system}\n"
            f"Para responder debés invocar la herramienta '{FINAL_RESULT_TOOL_NAME}' "
            "con argumentos que respeten su esquema. No respondas con texto libre "
            "ni inventes campos que no estén en el esquema."
        )


def _validate_final_result(call: ToolCall, schema: Any) -> Any:
    """Parsea y valida los `arguments` de un tool_call a `final_result`."""
    raw = call.arguments or "{}"
    arguments = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(arguments, dict):
        raise ValueError(
            f"los argumentos de {FINAL_RESULT_TOOL_NAME} deben ser un objeto JSON, "
            f"llegó {type(arguments).__name__}"
        )
    return schema.model_validate(arguments)


def _repair_prompt(detail: str) -> str:
    """Mensaje de reparación con el motivo concreto del rechazo."""
    return (
        f"La respuesta anterior fue rechazada porque {detail}. "
        f"Volvé a intentarlo invocando '{FINAL_RESULT_TOOL_NAME}' con todos los "
        "campos requeridos y con los tipos exactos que pide el esquema."
    )


def _failed_step(call: ToolCall, error: str) -> tuple[AgentStep, str]:
    """Paso fallido + el texto que se le devuelve al LLM como resultado."""
    return (
        AgentStep(
            tool_name=call.name,
            tool_input=call.arguments,
            tool_output=None,
            error=error,
        ),
        error,
    )


def _is_transient(exc: BaseException) -> bool:
    """¿El fallo parece transitorio y vale la pena reintentar?"""
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    huella = f"{type(exc).__name__} {exc}".lower()
    return any(marker in huella for marker in _TRANSIENT_MARKERS)


def _last_user_index(messages: list[dict[str, Any]]) -> int | None:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return None


def _drop_from_first_user(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Descarta lo que quedó antes del primer mensaje `user` de la ventana.

    Bedrock Converse rechaza con "A conversation must start with a user
    message" cualquier ventana que arranque en `assistant` o en `tool`,
    que es justo lo que produce el recorte por la izquierda. El
    `MockLLMClient` no se queja, así que esto no aparece hasta que se
    corre contra el proveedor real.
    """
    for i, message in enumerate(messages):
        if message.get("role") == "user":
            return messages[i:]
    return []


def _drop_orphan_tool_results(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Quita los `role: "tool"` cuyo `assistant` quedó fuera de la ventana.

    Un resultado de herramienta sin la llamada que lo originó rompe a los
    proveedores reales (Bedrock exige que cada `toolResult` siga a su
    `toolUse`) y no le dice nada al modelo. Recortar por la izquierda es
    justamente lo que produce esos huérfanos, así que se limpian acá.
    """
    out: list[dict[str, Any]] = []
    hay_llamada_abierta = False
    for message in messages:
        if message.get("role") == "tool":
            if hay_llamada_abierta:
                out.append(message)
            continue
        hay_llamada_abierta = bool(
            message.get("role") == "assistant" and message.get("tool_calls")
        )
        out.append(message)
    return out


def _prune(messages: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    """Recorta el historial retenido a lo que puede llegar a enviarse.

    Guardar más mensajes de los que caben en una ventana solo consume
    memoria: nunca se enviarían. El recorte se hace por la cola y se
    limpian los huérfanos que deja el corte.
    """
    if len(messages) <= budget:
        return list(messages)
    return _drop_orphan_tool_results(messages[-budget:])


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
