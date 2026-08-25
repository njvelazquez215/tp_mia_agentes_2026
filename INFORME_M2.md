# Informe — Milestone 2: Memoria, prompting y robustez

**Materia:** MIA303 — Agentes Autónomos y Sistemas de Decisión
**Entrega:** Milestone 2

---

## Qué cambió respecto del M1

La fachada externa quedó intacta: `build_agent`, `register_tool` y `run` se usan igual que en el M1, y el cliente LLM (`mia_agents/llm_client.py`) no se tocó. Todo lo que agregamos vive dentro del agente, así que se puede ejercitar inyectando un `MockLLMClient`.

```
student_framework/
  agent.py                  estado conversacional, ventana de historial,
                            structured_call con reparación, reintentos, tokens
  __init__.py               build_agent propaga las claves opcionales de config
  tools/calculator.py       errores recuperables con mensaje accionable
  tools/file_reader.py      sandbox de rutas + listado del directorio al fallar
tests/
  test_student_scenarios_m2.py   27 escenarios propios de M2
scripts/
  chat.py                   consola interactiva para probar el agente a mano
  m2_smoke.py               smoke test contra el proveedor real
```

También agregamos `.env` al `.gitignore` (no estaba).

El bucle de `run` sigue siendo el del M1; lo que cambia es de dónde salen los mensajes que se le mandan al LLM y qué pasa alrededor de cada llamada:

```
run(user_message)
     │
     ├── self._history += user_message          ← estado entre llamadas a run
     │
     ▼
  working = copia del historial
     │
     ├──────────────► _window(working) ─────► ventana ≤ max_history_messages
     │                                        (cola + ancla del último user,
     │                                         sin huérfanos, arranca en user)
     ▼
  _chat(ventana)  ── reintenta timeouts/5xx/throttling ──► LLMClient.chat(...)
     │                                                          │
     │                                                  LLMResponse (content,
     │                                                  tool_calls, tokens)
     ▼
  ¿tool_calls?
     │  NO ──► answer = content ──► fin del turno
     │  SÍ ──► ejecutar cada tool (con reintentos)
     │         AgentStep + mensaje role:"tool" ──► volver al chat
     ▼
  self._history = _prune(working)               ← se consolida al final
  AgentResult(answer, steps, error, input_tokens, output_tokens)
```

---

## 1. Estrategia de memoria

### El estado

`MyAgent` guarda la conversación en `self._history`. Cada `run` apila el mensaje del usuario sobre ese historial y después trabaja sobre una copia (`working`); el historial recién se pisa cuando el turno termina. Lo hicimos así a propósito: si el turno se corta a la mitad (por ejemplo, un error del proveedor que no es transitorio y se propaga), el historial no queda con un `assistant` que pidió herramientas y ningún resultado detrás. También agregamos `reset()`, que descarta el estado sin tener que reconstruir el agente.

### La ventana

La estrategia es ventana deslizante sobre la cola, con el último mensaje del usuario anclado. Está en `MyAgent._window`:

```python
budget = self._max_history_messages
cola = messages[-budget:] if len(messages) > budget else list(messages)
anchor = _last_user_index(messages)

if anchor is None:
    return _drop_orphan_tool_results(cola)

pedido = messages[anchor]
if any(m is pedido for m in cola):
    return _drop_from_first_user(_drop_orphan_tool_results(cola))

return [pedido] + _drop_orphan_tool_results(cola[1:])
```

El criterio de qué se conserva sale de cómo funciona el bucle:

- **La cola**, porque en un bucle agente lo más reciente es lo más útil. Los resultados de las herramientas del turno en curso son justo lo que el modelo necesita para redactar la respuesta; el contexto de diez turnos atrás casi nunca lo es.
- **El último mensaje del usuario, siempre.** Es la invariante de recencia que pide el enunciado. Con presupuestos chicos y turnos que encadenan varias herramientas, el pedido original se cae de la cola; cuando eso pasa lo ponemos al frente y cedemos un lugar de la cola para no pasarnos del tope. Sin ese anclaje el modelo terminaba viendo solo resultados de herramientas y ninguna pregunta.

El system prompt no gasta presupuesto porque viaja en el parámetro `system=` de `chat`, no en la lista de mensajes. Y el historial retenido se poda con `_prune` al mismo tope: guardar más mensajes de los que alguna vez van a entrar en una ventana solo ocupa memoria.

### Los problemas que aparecieron

**Los resultados de herramienta huérfanos.** Cortar por la izquierda deja mensajes `role: "tool"` cuyo `assistant` con el `tool_call` quedó afuera. Al modelo no le dicen nada y a Bedrock lo rompen directamente, porque la API Converse exige que cada `toolResult` siga a su `toolUse`. Lo limpia `_drop_orphan_tool_results`.

**La ventana se recalcula en cada vuelta, no una vez por turno.** El primer intento aplicaba el recorte al entrar a `run`, pero dentro de un mismo turno el historial sigue creciendo (un `assistant` más N `tool` por iteración) y se pasaba del tope en la segunda vuelta. Ahora `_window` se aplica en cada llamada a `chat`.

**Presupuestos degenerados.** Con `max_history_messages=1` la ventana queda con un único mensaje, el del usuario. El constructor fuerza `max(1, ...)` para que un 0 o un negativo no devuelvan una lista vacía.

Los dos que siguen no los encontraron los tests con mock, sino correr `scripts/m2_smoke.py` contra Bedrock. Vale la pena contarlos porque explican por qué la ventana quedó como quedó.

**La ventana tiene que empezar con un mensaje `user`.** En cuanto la conversación pasa el presupuesto, el corte por la izquierda la deja arrancando en `assistant`, y Bedrock responde `ValidationException: A conversation must start with a user message`. El `MockLLMClient` acepta cualquier lista, así que esto no se ve hasta que hay un proveedor real del otro lado. Lo resuelve `_drop_from_first_user`.

**Anclar el pedido no puede costar el resto de la ventana.** El primer arreglo del punto anterior aplicaba el "cortar hasta el primer `user`" también a la cola de la rama de anclaje. Esa cola son los `assistant` y `tool` del turno en curso, o sea que no contiene ningún `user`: la ventana colapsaba a un solo mensaje. El modelo dejaba de ver los resultados de las herramientas que él mismo acababa de pedir y las volvía a llamar. Contra Bedrock se veía como el agente invocando diez veces seguidas la calculadora en círculo; después del arreglo, el mismo pedido se resuelve en dos llamadas. Fue el bug más caro que
tuvimos y ninguno de los tests lo mostraba, porque todos miraban el tamaño de la ventana y ninguno su contenido. Ahora hay dos que sí lo miran (`test_la_ventana_conserva_el_contexto_del_turno_en_curso` y
`test_la_ventana_siempre_empieza_con_un_mensaje_de_usuario`).

### Tradeoffs

A favor de la ventana deslizante: cuesta O(n) por llamada, no agrega ninguna llamada extra al LLM y es determinista, así que se puede testear entera con el mock. En contra: se olvida. Un dato mencionado treinta turnos atrás se pierde aunque siga siendo relevante, y no hay forma de recuperarlo.

Evaluamos las dos alternativas que menciona el enunciado y las descartamos. *Summarization* (comprimir lo viejo en un mensaje de resumen) implica una llamada extra al LLM por compactación: costo, latencia y una fuente nueva de alucinación sobre la que después el modelo razona como si fuera un hecho. *Offload/retrieve* (guardar los mensajes afuera y recuperarlos por similitud) necesita embeddings o un índice, que es bastante más infraestructura de la que este milestone pide. Como la ventana deslizante es la estrategia obligatoria y las otras dos son opcionales, preferimos que la obligatoria estuviera sólida
en los casos borde antes que sumar una segunda a medias.

---

## 2. Salida estructurada

### Cómo se le ofrece `final_result` al LLM

`structured_call` deriva la herramienta sintética del propio schema de Pydantic y la ofrece como única herramienta en todas las llamadas, también en las de reparación:

```python
tool = final_result_tool_schema(schema)      # name == FINAL_RESULT_TOOL_NAME
resp = self._llm.chat(messages=self._window(messages), tools=[tool], system=...)
```

No se registra con `register_tool` porque no es una herramienta del agente sino un mecanismo de cierre, y no se mezcla con las herramientas que expone `run`. El system prompt del método suma la instrucción explícita de invocar `final_result` y de no contestar con texto libre ni inventar campos fuera del esquema.

El método corre sobre su propio hilo de mensajes, aislado de la conversación: los intentos fallidos y los prompts de reparación no se mezclan con el historial de `run`.

### Cómo se validan los argumentos

En `_validate_final_result`, en tres pasos, cada uno con su modo de fallo:

1. `json.loads(call.arguments)` levanta `JSONDecodeError` si el modelo emitió JSON truncado o mal formado.
2. Se chequea que el resultado sea un objeto; si mandó una lista o un escalar, `ValueError`.
3. `schema.model_validate(arguments)` levanta `ValidationError` si faltan campos requeridos o los tipos no coinciden, por ejemplo `"resultado": "cuarenta y dos"` contra un `int`.

### Cómo se reparan los fallos

Hay dos rutas según lo que haya pasado. Si el modelo contestó texto libre sin invocar la tool, se agrega su mensaje `assistant` y un `user` de reparación. Si el `tool_call` llegó pero no validó, se agrega el `assistant` con el `tool_call`, un mensaje `role: "tool"` con `"Validación fallida: <detalle>"` y después el `user` de reparación. Que el fallo viaje también como mensaje `tool` mantiene la conversación bien formada para los proveedores reales, donde cada `toolUse` necesita su `toolResult`.

El prompt de reparación siempre incluye el motivo concreto (el mensaje de Pydantic o del parser, no un genérico), que es lo que le permite al modelo corregir el campo puntual:

```python
def _repair_prompt(detail: str) -> str:
    return (
        f"La respuesta anterior fue rechazada porque {detail}. "
        f"Volvé a intentarlo invocando '{FINAL_RESULT_TOOL_NAME}' con todos los "
        "campos requeridos y con los tipos exactos que pide el esquema."
    )
```

### Qué pasa cuando se agotan los reintentos

Se hacen como mucho `1 + max_repair_attempts` llamadas, o sea tres con el valor por defecto. Agotadas, `structured_call` levanta `StructuredOutputError` con el nombre del schema, la cantidad de intentos y el último error de validación. No devuelve `None`, no devuelve una instancia a medias y no sigue reintentando: el contrato observable es instancia válida o excepción, y quien llama decide qué hacer con el fallo.

---

## 3. Errores en las herramientas

El criterio para separar recuperable de no recuperable fue: **es recuperable si el LLM lo puede arreglar cambiando los argumentos**. Todo lo recuperable se devuelve como un `str` que empieza con `"Error:"`, que para el bucle es un resultado normal de la herramienta: el modelo lo lee y reintenta. Lo no recuperable se deja escapar como excepción y el agente lo registra en `AgentStep.error`.

### Calculadora

| Error recuperable | Qué devuelve |
|---|---|
| Operando no numérico | qué parámetro falló, qué valor recibió, de qué tipo era y un ejemplo de llamada bien formada |
| Operador no soportado | el operador rechazado y la lista completa de válidos (`+ - * / %`) |
| División por cero | que `right_operand` no puede ser 0 con `/`, y qué alternativa hay |
| Módulo por cero | que el resto de una división por 0 no está definido |

`_as_number` acepta además números que vienen como string (`"17"`, `"17,5"`), que es frecuente cuando el modelo serializa los argumentos. Es un falso error y no tiene sentido hacérselo corregir.

Ejemplo de recuperación, en `test_recuperacion_calculadora_en_el_bucle`: el modelo pide `operator="^"`, la herramienta devuelve

```
Error: operador no soportado '^'. Usá exactamente uno de estos: +, -, *, /, %.
```

el agente lo vuelca como mensaje `role: "tool"` y en la vuelta siguiente el modelo llama con `operator="%"` y obtiene `1`. El `AgentResult` queda con dos steps, el primero con el error y el segundo exitoso.

### Lector de archivos

Agregamos un sandbox: la raíz es el directorio de trabajo, o `MIA_FILE_SANDBOX` si está definida. Es un cambio de comportamiento respecto del M1, donde no había aislamiento de rutas (era la limitación 4 de aquel informe).

| Error recuperable | Qué devuelve |
|---|---|
| Ruta vacía | que está vacía, más la regla de cómo se ve una ruta válida |
| Ruta absoluta | que solo se admiten relativas, cuál es el sandbox y la regla |
| Ruta con `..` | que no se puede subir de directorio, y la regla |
| Ruta que escapa del sandbox al resolver | que apunta afuera, y la regla |
| Archivo inexistente | si el directorio contenedor existe, el listado de sus archivos (hasta 40, con `/` marcando subdirectorios) para que el modelo elija el nombre correcto |
| La ruta es un directorio | que es un directorio, y su contenido listado |
| Binario o demasiado grande | qué restricción se violó (UTF-8, 100 KB) |

El listado omite las entradas ocultas (`.env`, `.git/`, `.venv/`): no son lo que el modelo busca y solo meten ruido, y nombres sensibles, en el contexto. Lo notamos leyendo la salida del smoke test, donde el listado real arrancaba con `.env`.

Ejemplo de recuperación, en `test_recuperacion_lector_en_el_bucle`: el modelo pide `informes.txt`, que no existe, y recibe

```
Error: el archivo 'informes.txt' no existe. En '.' hay: informe.txt. Elegí uno de esos nombres y reintentá.
```

En la vuelta siguiente pide `informe.txt` y obtiene el contenido. Contra Bedrock el mismo flujo con `requirement.txt` → `requirements.txt` se resolvió en dos vueltas, aunque no siempre: el modelo a veces prefiere contestar con el listado en lugar de reintentar. La herramienta cumple su parte dándole la información; que la aproveche depende del modelo.

### Errores del agente al invocar herramientas

Aprovechamos para mejorar los mensajes de `_execute_tool_call`, que es la otra fuente de errores que el modelo puede corregir: JSON de argumentos mal formado (se le pide un objeto JSON con los parámetros del esquema), herramienta alucinada (se le lista qué herramientas existen) y `TypeError` porque los argumentos no encajan en la firma (se le listan los parámetros esperados, leídos del `ToolSchema`).

---

## Resiliencia y conteo de tokens

`_with_retries` envuelve tanto las llamadas al LLM como la ejecución de cada herramienta. Reintenta `max_retries` veces (2 por defecto) con backoff lineal, y solo si el fallo parece transitorio:

```python
def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    huella = f"{type(exc).__name__} {exc}".lower()
    return any(marker in huella for marker in _TRANSIENT_MARKERS)
```

La clasificación mira el tipo y, si no alcanza, el nombre de la clase y el mensaje contra una lista de marcas (`timeout`, `throttl`, `rate limit`, `429`, `500`/`502`/`503`/`504`, `service unavailable`, `connection reset`, etc.). Es por texto y no por jerarquía de excepciones a propósito: cada SDK trae la suya (`botocore.ThrottlingException`, `httpx.ReadTimeout`, `ollama.ResponseError`) y el agente no debería importar ninguno para seguir siendo agnóstico del proveedor.

Todo lo demás se propaga sin envolver. Un `ValueError("modelo inexistente")` sale tal cual y sin reintentos, porque reintentar un error de configuración solo agrega latencia.

Los tokens se acumulan con `_accumulate` sobre todas las llamadas del turno: mantiene `None` hasta que llega el primer valor real y a partir de ahí trata los `None` posteriores como 0, que es el contrato de `AgentResult`. Los contadores son locales al turno, no se arrastran entre llamadas a `run`. Y si el modelo cierra con `content` vacío, o si se agotan las iteraciones, `run` devuelve un texto de último recurso en lugar de `""`, para que `answer` nunca quede vacío.

---

## 4. Modos de fallo dentro y fuera del alcance

### Dentro

1. Conversaciones largas que superan el presupuesto de contexto.
2. Resultados de herramienta huérfanos y ventanas que no arrancan en `user`.
3. Modelo que contesta texto libre cuando se le pidió salida estructurada.
4. Argumentos de `final_result` con JSON roto, campos faltantes o tipos incorrectos.
5. Argumentos de herramientas con JSON roto o incompatibles con la firma.
6. Herramientas alucinadas.
7. Excepciones lanzadas por un callable de herramienta.
8. Timeouts, 5xx, throttling y errores de red, tanto del LLM como de las herramientas.
9. Loops de tool calls que nunca cierran (tope de `max_iterations`).
10. Usos inválidos de la calculadora y del lector, con mensajes accionables.

### Fuera, y por qué

**Presupuesto por tokens.** El tope es por cantidad de mensajes, que es lo que pide el enunciado. Un mensaje de 50.000 caracteres cuenta igual que uno de 10, así que la ventana no garantiza entrar en la ventana de contexto del modelo. Hacerlo por tokens exigiría un tokenizador por proveedor.

**Recuperación del contexto descartado.** No hay resumen ni retrieval: lo que sale de la ventana se pierde. Es el tradeoff explícito de la sección 1.

**Reintentos con jitter, presupuesto global de reintentos o circuit breaker.** El backoff es lineal y por llamada. Alcanza para fallos transitorios puntuales, no para una caída sostenida del proveedor.

**Timeout propio por herramienta.** Si un callable se cuelga, el agente se cuelga con él: no hay watchdog ni ejecución en un thread aparte. Tampoco hay paralelismo entre los `tool_calls` de un mismo turno, se ejecutan en orden.

**Validación semántica.** Que `final_result` valide contra el schema no dice nada sobre si el contenido es correcto. El agente valida forma, no verdad.

**Concurrencia.** `MyAgent` no es thread-safe: dos `run` simultáneos sobre la misma instancia se pisarían el historial.

**Sandbox a prueba de adversarios.** El lector bloquea rutas absolutas, `..` y escapes por resolución de symlinks, pero está pensado contra errores del modelo, no contra alguien con control del filesystem (TOCTOU, links creados entre la validación y la lectura).

**`structured_call` con herramientas.** Solo se ofrece `final_result`; no hay un bucle de herramientas que termine en salida estructurada. El enunciado pide esa tool como única forma de cierre del método.

---

## Cómo ejecutarlo

### Tests

```bash
# (Windows PowerShell, desde la raíz del repo)
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Conformidad M1 + M2
.\.venv\Scripts\python.exe -m pytest tests/conformance/test_m1.py tests/conformance/test_m2.py -v

# Escenarios propios
.\.venv\Scripts\python.exe -m pytest tests/test_student_scenarios.py tests/test_student_scenarios_m2.py -v

# Todo junto (M3 todavía no está en el repo: mia_world/ no existe)
.\.venv\Scripts\python.exe -m pytest --ignore=tests/conformance/test_m3_world.py -q
```

Pasan 91 tests: 5 de conformidad de M1, 7 de conformidad de M2, 11 escenarios propios de M1, 27 de M2 y 41 de los providers y el schema que provee la cátedra. Todos usan `MockLLMClient`, así que son deterministas y no consumen API.

### Contra un LLM real

La configuración va en un `.env`. Probamos con AWS Bedrock, modelo `amazon.nova-lite-v1:0` en `us-east-2`.

```bash
# Una sola corrida, salida JSON
python -m mia_agents.cli run --module student_framework --message "¿Cuánto es 17 * 23? Usá la calculadora."

# Smoke test de M2 de punta a punta
.\.venv\Scripts\python.exe scripts/m2_smoke.py

# Consola interactiva
.\.venv\Scripts\python.exe scripts/chat.py --presupuesto 8
```

El smoke test verifica en vivo lo que el mock no puede: cinco turnos con presupuesto de 8 mensajes sin pasarse del tope y sin `answer` vacío, statefulness con presupuesto holgado (el modelo recupera un dato dado tres turnos antes), la recuperación de la calculadora y del lector, y un `structured_call` que devuelve la instancia de Pydantic validada.

La consola sirve para probarlo a mano. Además de charlar acepta `/debug`, que imprime la ventana exacta que se le manda al modelo en cada llamada —la forma más directa de ver el recorte funcionando—, `/historial`, `/tokens`, `/reset` y `/estructurado <pedido>`.
