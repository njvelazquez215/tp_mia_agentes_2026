# Informe — Milestone 1: Bucle del agente y herramientas

**Materia:** MIA303 — Agentes Autónomos y Sistemas de Decisión
**Entrega:** Milestone 1

---

## 1. Diagrama de arquitectura

El framework separa lo provisto por la cátedra, en `mia_agents/` de lo realizado para la entrega (`student_framework/`). La única puerta de entrada pública es `build_agent`, que tanto la CLI como los tests de conformidad invocan.

```
                          ┌──────────────────────────────────────┐
   usuario / test         │            student_framework/        │
       │                  │                                      │
       │  user_message    │   build_agent(config) ──► MyAgent    │
       ▼                  │        │                    │  ▲     │
 ┌───────────┐  run(msg)  │        │ register_tool(fn,   │  │     │
 │  CLI /    │───────────►│        │      schema) x3     │  │     │
 │  pytest   │            │        ▼                     │  │     │
 └───────────┘            │   ┌──────────────┐  _tools{} │  │     │
       ▲                  │   │ tools/       │  _schemas{}│ │     │
       │  AgentResult     │   │ calculator   │           │  │     │
       └──────────────────┼── │ file_reader  │           │  │     │
                          │   │ word_counter │           │  │     │
                          │   └──────────────┘           │  │     │
                          └──────────────────────────────┼──┼─────┘
                                                         │  │
                          ┌──────────────────────────────▼──┴─────┐
       bucle run():       │            MyAgent.run()              │
                          │                                       │
   ┌──────────────────────┤  1. messages = [user_message]        │
   │                      │  2. resp = self._llm.chat(            │
   │                      │        messages,                      │
   │   ┌──────────────┐   │        tools=list(_schemas.values()), │
   │   │  LLMClient   │◄──┤        system)                        │
   │   │ (protocolo   │   │  3. ¿resp.tool_calls?                 │
   │   │  chat(...))  │──►│       NO ► answer = resp.content ─► fin│
   │   └──────┬───────┘   │       SÍ ► por cada tool_call:        │
   │         │            │            ejecutar callable,         │
   │   ┌──────▼───────┐   │            AgentStep, msg role:"tool" │
   │   │ Bedrock /    │   │  4. repetir hasta max_iterations      │
   │   │ Ollama /     │   └───────────────────────────────────────┘
   │   │ MockLLMClient│                     │
   │   └──────────────┘                     ▼
   │                                  AgentResult(answer, steps,
   └─ (inyectado vía config["llm_client"])  input/output_tokens, error)
```

**Lectura del diagrama:**

- `build_agent` instancia `MyAgent`, le inyecta el `LLMClient` (real desde el entorno o el `MockLLMClient` que pasan los tests vía `config["llm_client"]`) y le registra las tres herramientas.
- `MyAgent` guarda las herramientas en dos diccionarios (`_tools`, `_schemas`) y corre el bucle agente en `run`.
- En cada vuelta llama al LLM solo a través del protocolo `chat(...)`: el agente no sabe si detrás hay Bedrock, Ollama o un mock. Eso es lo que hace la implementación evaluable de forma determinista.

---

## 2. Diseño de la interfaz de herramientas

Toda la cadena va de la firma Python al JSON Schema, sin escribir JSON a mano.

### a) Definición de la herramienta (`student_framework/tools/`)

Cada herramienta es un `callable` con tipos en la firma, `Annotated[..., Field(...)]` para describir cada argumento, y un docstring que describe la herramienta. Ejemplo real (la calculadora):

```python
def calculator(
    left_operand:  Annotated[float, Field(description="Primer operando…")],
    right_operand: Annotated[float, Field(description="Segundo operando…")],
    operator:      Annotated[str,   Field(description="Uno de: +, -, *, /, %")],
) -> str:
    """Realiza una única operación aritmética binaria entre dos números."""
    ...

calculator_schema = ToolSchema.from_callable(calculator)
```

- El docstring → `ToolSchema.description` (lo que el LLM lee para decidir si usar la tool).
- Los tipos + `Field(description=...)` → `ToolSchema.parameters` (el JSON Schema de argumentos). `ToolSchema.from_callable` construye internamente un modelo Pydantic desde la firma y exporta su `model_json_schema()`.

### b) Qué se guarda en `register_tool`

```python
def register_tool(self, tool: Callable[..., str], schema: ToolSchema) -> None:
    self._tools[schema.name]   = tool     # el callable a ejecutar
    self._schemas[schema.name] = schema   # el ToolSchema a exponer al LLM
```

Se guardan dos cosas indexadas por `schema.name`:

| Diccionario | Contenido | Para qué |
|---|---|---|
| `self._tools`   | el `callable` | ejecutar la herramienta cuando el LLM la pide |
| `self._schemas` | el `ToolSchema` | exponerla al LLM en `chat(tools=...)` |

Usar `schema.name` como clave garantiza que el `tool_name` que devuelve el LLM en
un `tool_call` mapea exactamente al callable correcto y al nombre del esquema.

### c) Qué se pasa a `chat(tools=...)`

En cada vuelta del bucle:

```python
resp = self._llm.chat(
    messages=messages,
    tools=list(self._schemas.values()) if self._schemas else None,
    system=self._system,
)
```

Se pasa la lista de objetos `ToolSchema` (no dicts). Si no hay herramientas registradas, se pasa `None` (no una lista vacía). El contrato exige que en la primera llamada, si hay tools registradas, `tools` **no** sea `None` y que el `schema.name` aparezca en esa lista — eso se cumple por construcción.

### d) Qué hace el `LLMClient` fijo con cada esquema

El `LLMClient` (FIJO, `mia_agents/llm_client.py`) traduce cada `ToolSchema` al formato nativo del proveedor:

1. `_tool_specs_as_dicts` aplica `ToolSchema.to_llm_spec()` → `{name, description, parameters}`.
2. `_wrap_tool_spec` lo envuelve según el proveedor:
   - **Ollama:** `{"type": "function", "function": {name, description, parameters}}`.
   - **Bedrock (Converse):** `{"toolSpec": {name, description, inputSchema: {json: parameters}}}`.

El agente nunca toca ese formato: entrega `ToolSchema` y recibe un `LLMResponse`
normalizado (`content`, `tool_calls`, `input_tokens`, `output_tokens`).

---

## 3. Cómo termina el bucle y qué pasa al alcanzar el límite

El bucle de `run` está acotado por `max_iterations` (default 10) y puede terminar de dos formas:

### a) Terminación normal — respuesta final (condición de parada M1)

Cuando el LLM responde texto sin `tool_calls`, ese texto es la respuesta final:

```python
if not resp.tool_calls:
    return AgentResult(answer=resp.content or "", steps=steps, ...)
```

- Si nunca hubo herramientas → `steps == []` y el LLM se llamó una sola vez.
- Si hubo herramientas → cada invocación dejó un `AgentStep` en `steps`.

### b) Terminación por límite — `max_iterations`

Si el LLM entra en un loop pidiendo herramientas sin cerrar nunca, el `for` recorre como mucho `max_iterations` vueltas y deja de llamar al LLM. Aun así `run` devuelve un `AgentResult` válido, con:

- `answer` = último `content` visto (o `""`),
- `steps` con todo lo ejecutado hasta el corte,
- `error` = `"Se alcanzó el límite de N iteraciones."`

Esto garantiza que el agente **nunca cuelga** ni lanza excepción por loops infinitos (ver test `test_corta_por_max_iterations`: con 50 respuestas en loop, el mock se llama exactamente 10 veces).

### c) Realimentación entre vueltas

Cuando hay `tool_calls`, antes de volver a llamar al LLM el agente:

1. Agrega un mensaje `assistant` con los `tool_calls` al historial.
2. Ejecuta cada callable y agrega un mensaje `role: "tool"` con su salida.

Así, el valor devuelto por la herramienta aparece en los `messages` de la siguiente llamada a `chat`, que es justo lo que el contrato pide.

---

## 4. Limitaciones conocidas

1. **Sin estado entre llamadas a `run` (por diseño de M1).** Cada `run` arranca con un historial nuevo (`messages = [user_message]`). La conversación multiturno y el respeto de `max_history_messages` son de M2; el constructor ya acepta el parámetro pero el bucle de M1 lo ignora.

2. **`structured_call` no implementado.** Queda como `NotImplementedError` (es el contrato de M2: tool sintética `final_result`, validación y reparación).

3. **Discrepancia de operadores en la calculadora.** El `ENUNCIADO_M1.md` pide `+ - * %` (módulo) y el PDF del campus pide `+ - * /` (división). Para no depender de cuál se evalúe, la herramienta **soporta los cinco** (`+ - * / %`). Los casos degenerados (división/módulo por cero, operador inválido) devuelven un mensaje de error legible como `str` en lugar de lanzar excepción, para que el LLM pueda observarlo y reaccionar.

4. **Lector de archivos: solo texto UTF-8 y con tope de tamaño.** Rechaza binarios (`UnicodeDecodeError`), directorios y archivos inexistentes devolviendo un mensaje de error (no excepción). Hay un límite defensivo de 100 KB para no volcar archivos enormes al contexto del LLM. **No hay sandbox de rutas:** la herramienta puede leer cualquier archivo de texto al que el proceso tenga acceso (la consigna pide E/S "restringida" a texto, no aislamiento de rutas).

5. **Ejecución secuencial de tool_calls.** Si el LLM emite varios `tool_calls` en un mismo turno, se ejecutan en orden, de forma síncrona. No hay paralelismo ni timeouts por herramienta.

6. **Tokens dependientes del proveedor.** `input_tokens`/`output_tokens` se acumulan solo si el `LLMResponse` los reporta; con el `MockLLMClient` sin tokens programados quedan en `None` (comportamiento esperado por el contrato).

7. **Robustez ante el LLM, no validación semántica.** El agente maneja JSON de argumentos malformado, herramientas alucinadas y excepciones de los callables (todos quedan como `AgentStep` con `error` no nulo), pero **no valida** que los argumentos tengan sentido más allá de lo que haga el propio callable.

---

## Apéndice — Cómo ejecutar

```bash
# (Windows PowerShell, desde la raíz del repo)
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Tests de conformidad M1 + escenarios propios (deterministas, sin API)
.\.venv\Scripts\python.exe -m pytest tests/conformance/test_m1.py tests/test_student_scenarios.py -v

# Ejecutar el agente contra un LLM real (requiere proveedor configurado:
# OLLAMA_HOST o BEDROCK_MODEL_ID). En PowerShell, todo en una sola línea:
python -m mia_agents.cli run --module student_framework --message "¿Cuánto es 17 * 23? Usá la calculadora."
```

Resultado actual: 16 tests pasan (5 de conformidad M1 + 11 propios).
