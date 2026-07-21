"""Escenarios propios del M2: memoria, reparación, resiliencia y tokens.

Complementan `tests/test_student_scenarios.py` (M1) y van más allá de los
tests de conformidad: acá probamos los casos límite que el enunciado pide
demostrar (conversación que revienta el presupuesto, prompt estructurado
deliberadamente roto, timeout simulado, errores accionables de las tools).

Todo corre contra el `MockLLMClient`, sin API ni red.
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from mia_agents.testing import MockLLMClient
from mia_agents.tool_schema import FINAL_RESULT_TOOL_NAME
from mia_agents.types import LLMResponse, ToolCall

from student_framework import build_agent
from student_framework.agent import StructuredOutputError
from student_framework.tools.calculator import calculator
from student_framework.tools.file_reader import read_text_file


def _tc(name: str, args: dict, call_id: str = "c1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=json.dumps(args))


def _final_result(args: dict | str, call_id: str = "fr") -> LLMResponse:
    if isinstance(args, dict):
        args = json.dumps(args)
    return LLMResponse(
        content=None,
        tool_calls=[ToolCall(id=call_id, name=FINAL_RESULT_TOOL_NAME, arguments=args)],
    )


def _agente(mock: MockLLMClient, **config):
    # retry_backoff=0 para no dormir en los tests de reintentos.
    return build_agent({"llm_client": mock, "retry_backoff": 0.0, **config})


# --------------------------------------------------------------------------
# Memoria: conversación larga contra un presupuesto chico.
# --------------------------------------------------------------------------


def test_conversacion_larga_respeta_presupuesto_y_recencia() -> None:
    """30 turnos con mensajes grandes: nunca se pasa del tope ni pierde el
    último pedido del usuario, y toda corrida devuelve answer no vacío."""
    budget = 8
    turnos = 30
    mock = MockLLMClient(
        [LLMResponse(content=f"respuesta {i}") for i in range(turnos)]
    )
    agent = _agente(mock, max_history_messages=budget)

    for i in range(turnos):
        pedido = f"turno {i}: " + ("texto largo de relleno " * 200)
        result = agent.run(pedido)

        assert result.answer, "cada run debe devolver un answer no vacío"
        enviados = mock.calls[-1]["messages"]
        assert len(enviados) <= budget
        # Invariante de recencia: lo que el usuario acaba de decir viaja sí o sí.
        assert any(f"turno {i}:" in str(m.get("content", "")) for m in enviados)


def test_historial_no_crece_sin_limite() -> None:
    budget = 6
    mock = MockLLMClient([LLMResponse(content="ok") for _ in range(40)])
    agent = _agente(mock, max_history_messages=budget)

    for i in range(40):
        agent.run(f"turno {i}")

    assert len(agent.history) <= budget


def test_ventana_no_deja_resultados_de_tool_huerfanos() -> None:
    """Recortar por la izquierda no puede dejar un `tool` sin su assistant."""
    budget = 4
    respuestas: list[LLMResponse] = []
    for i in range(6):
        respuestas.append(
            LLMResponse(
                content=None,
                tool_calls=[
                    _tc(
                        "calculator",
                        {"left_operand": i, "right_operand": 1, "operator": "+"},
                        f"c{i}",
                    )
                ],
            )
        )
    respuestas.append(LLMResponse(content="listo"))

    mock = MockLLMClient(respuestas)
    agent = _agente(mock, max_history_messages=budget, max_iterations=7)
    result = agent.run("encadená varias cuentas")

    assert result.answer == "listo"
    for call in mock.calls:
        mensajes = call["messages"]
        assert len(mensajes) <= budget
        assert mensajes[0].get("role") == "user"
        for anterior, actual in zip(mensajes, mensajes[1:]):
            if actual.get("role") == "tool":
                assert anterior.get("role") in ("assistant", "tool")


def test_la_ventana_conserva_el_contexto_del_turno_en_curso() -> None:
    """Anclar el pedido del usuario no puede costar el resto de la ventana.

    Con el historial lleno y un turno que encadena varias herramientas, el
    ancla se pone al frente pero los resultados de las tools de este mismo
    turno tienen que seguir viajando: si la ventana colapsa al pedido
    solo, el modelo no ve lo que ya ejecutó y repite llamadas en círculo
    (lo vimos contra Bedrock antes de arreglarlo).
    """
    budget = 8
    previos: list[LLMResponse] = []
    for i in range(4):
        previos += [
            LLMResponse(
                content=None,
                tool_calls=[
                    _tc(
                        "calculator",
                        {"left_operand": i, "right_operand": 1, "operator": "+"},
                        f"p{i}",
                    )
                ],
            ),
            LLMResponse(content=f"listo {i}"),
        ]
    turno_final = [
        LLMResponse(
            content=None,
            tool_calls=[
                _tc(
                    "calculator",
                    {"left_operand": i, "right_operand": 2, "operator": "*"},
                    f"f{i}",
                )
            ],
        )
        for i in range(3)
    ] + [LLMResponse(content="fin")]

    mock = MockLLMClient(previos + turno_final)
    agent = _agente(mock, max_history_messages=budget)
    for i in range(4):
        agent.run(f"turno previo {i}")

    llamadas_previas = mock.call_count
    agent.run("encadená tres cuentas")

    ventanas = [c["messages"] for c in mock.calls[llamadas_previas:]]
    for n, mensajes in enumerate(ventanas):
        assert len(mensajes) <= budget
        assert mensajes[0]["role"] == "user"
        assert any("encadená tres cuentas" in str(m.get("content")) for m in mensajes)
        if n > 0:
            assert any(m["role"] == "tool" for m in mensajes), (
                "la ventana perdió los resultados de herramienta del turno actual"
            )


def test_la_ventana_siempre_empieza_con_un_mensaje_de_usuario() -> None:
    """Bedrock Converse rechaza toda conversación que no arranque en `user`.

    El recorte por la izquierda deja la ventana empezando en `assistant`
    en cuanto la conversación pasa el presupuesto, y el mock no se queja:
    esto lo encontramos corriendo contra el proveedor real.
    """
    budget = 6
    mock = MockLLMClient([LLMResponse(content=f"r{i}") for i in range(12)])
    agent = _agente(mock, max_history_messages=budget)

    for i in range(12):
        agent.run(f"turno {i}")

    for call in mock.calls:
        assert call["messages"][0]["role"] == "user", (
            f"la ventana empezó con {call['messages'][0]['role']!r}"
        )


def test_segundo_turno_ve_el_primero() -> None:
    mock = MockLLMClient(
        [LLMResponse(content="anotado"), LLMResponse(content="era ALFA-7")]
    )
    agent = _agente(mock)

    agent.run("guardá el código ALFA-7")
    agent.run("¿cuál era el código?")

    assert "ALFA-7" in str(mock.calls[1]["messages"])


def test_reset_borra_la_conversacion() -> None:
    mock = MockLLMClient([LLMResponse(content="a"), LLMResponse(content="b")])
    agent = _agente(mock)

    agent.run("primer mensaje")
    agent.reset()
    agent.run("segundo mensaje")

    assert "primer mensaje" not in str(mock.calls[1]["messages"])


# --------------------------------------------------------------------------
# Salida estructurada: reparación y fallo limpio.
# --------------------------------------------------------------------------


class Respuesta(BaseModel):
    resultado: int
    comentario: str


def test_structured_call_repara_texto_libre() -> None:
    """El prompt roto (el modelo contesta texto) dispara la reparación."""
    mock = MockLLMClient(
        [
            LLMResponse(content="El resultado es 42, obviamente."),
            _final_result({"resultado": 42, "comentario": "listo"}),
        ]
    )
    agent = _agente(mock)

    parsed = agent.structured_call(prompt="dame el resultado", schema=Respuesta)

    assert parsed.resultado == 42
    assert mock.call_count == 2
    # La tool sintética se ofrece en todas las llamadas, también en la de reparación.
    for call in mock.calls:
        assert [t.name for t in call["tools"]] == [FINAL_RESULT_TOOL_NAME]
    # El mensaje de reparación explica qué se rechazó.
    reparacion = str(mock.calls[1]["messages"])
    assert FINAL_RESULT_TOOL_NAME in reparacion


def test_structured_call_repara_json_malformado() -> None:
    mock = MockLLMClient(
        [
            _final_result('{"resultado": 42, "comentario":'),
            _final_result({"resultado": 42, "comentario": "ok"}, "fr2"),
        ]
    )
    agent = _agente(mock)

    parsed = agent.structured_call(prompt="dame el resultado", schema=Respuesta)

    assert parsed == Respuesta(resultado=42, comentario="ok")


def test_structured_call_falla_limpio_al_agotar_reintentos() -> None:
    mock = MockLLMClient(
        [
            _final_result({"comentario": "falta el resultado"}),
            _final_result({"resultado": "cuarenta y dos", "comentario": "x"}, "fr2"),
            LLMResponse(content="paso de las tools"),
        ]
    )
    agent = _agente(mock)

    with pytest.raises(StructuredOutputError):
        agent.structured_call(
            prompt="dame el resultado", schema=Respuesta, max_repair_attempts=2
        )

    assert mock.call_count == 3


def test_structured_call_no_contamina_la_conversacion() -> None:
    """La reparación es interna: no queda pegada al historial de `run`."""
    mock = MockLLMClient(
        [
            LLMResponse(content="hola"),
            _final_result({"resultado": 1, "comentario": "c"}),
            LLMResponse(content="chau"),
        ]
    )
    agent = _agente(mock)

    agent.run("hola")
    agent.structured_call(prompt="dame el resultado", schema=Respuesta)
    agent.run("chau")

    assert "dame el resultado" not in str(mock.calls[2]["messages"])


# --------------------------------------------------------------------------
# Resiliencia: fallos transitorios vs. errores reales.
# --------------------------------------------------------------------------


def test_timeout_del_llm_se_reintenta() -> None:
    mock = MockLLMClient(
        [
            TimeoutError("read timeout contra el proveedor"),
            LLMResponse(content="salió a la segunda", input_tokens=10, output_tokens=5),
        ]
    )
    agent = _agente(mock)

    result = agent.run("algo")

    assert result.answer == "salió a la segunda"
    assert result.input_tokens == 10
    assert mock.call_count == 2


def test_throttling_se_reintenta_hasta_el_tope() -> None:
    class ThrottlingException(Exception):
        pass

    mock = MockLLMClient(
        [
            ThrottlingException("Too Many Requests"),
            ThrottlingException("Too Many Requests"),
            LLMResponse(content="ok"),
        ]
    )
    agent = _agente(mock, max_retries=2)

    assert agent.run("algo").answer == "ok"
    assert mock.call_count == 3


def test_error_no_transitorio_aflora_limpio() -> None:
    mock = MockLLMClient([ValueError("modelo inexistente")])
    agent = _agente(mock)

    with pytest.raises(ValueError, match="modelo inexistente"):
        agent.run("algo")

    # No se reintenta: un error de configuración no mejora reintentando.
    assert mock.call_count == 1


def test_tool_con_fallo_transitorio_se_reintenta() -> None:
    intentos = {"n": 0}

    def tool_inestable(text: str) -> str:
        intentos["n"] += 1
        if intentos["n"] == 1:
            raise TimeoutError("la tool tardó demasiado")
        return f"ok: {text}"

    from mia_agents.types import ToolSchema

    mock = MockLLMClient(
        [
            LLMResponse(
                content=None, tool_calls=[_tc("tool_inestable", {"text": "hola"})]
            ),
            LLMResponse(content="hecho"),
        ]
    )
    agent = _agente(mock)
    agent.register_tool(tool_inestable, ToolSchema.from_callable(tool_inestable))
    result = agent.run("usá la tool")

    assert intentos["n"] == 2
    assert result.steps[0].tool_output == "ok: hola"
    assert result.steps[0].error is None


def test_tokens_se_cuentan_por_run_y_no_se_arrastran() -> None:
    mock = MockLLMClient(
        [
            LLMResponse(content="uno", input_tokens=100, output_tokens=10),
            LLMResponse(content="dos", input_tokens=50, output_tokens=5),
        ]
    )
    agent = _agente(mock)

    primero = agent.run("a")
    segundo = agent.run("b")

    assert (primero.input_tokens, primero.output_tokens) == (100, 10)
    assert (segundo.input_tokens, segundo.output_tokens) == (50, 5)


# --------------------------------------------------------------------------
# Errores recuperables de las herramientas.
# --------------------------------------------------------------------------


def test_calculadora_operando_no_numerico() -> None:
    out = calculator(
        left_operand="cuarenta y dos", right_operand=2, operator="+"
    )
    assert out.startswith("Error:")
    assert "left_operand" in out
    assert "cuarenta y dos" in out


def test_calculadora_acepta_numeros_como_texto() -> None:
    assert calculator(left_operand="17", right_operand="23", operator="*") == "391"


def test_calculadora_operador_invalido_lista_los_validos() -> None:
    out = calculator(left_operand=1, right_operand=2, operator="**")
    assert "no soportado" in out
    for op in ("+", "-", "*", "/", "%"):
        assert op in out


def test_calculadora_division_por_cero_explica_la_restriccion() -> None:
    out = calculator(left_operand=1, right_operand=0, operator="/")
    assert "right_operand" in out and "cero" in out


def test_lector_ruta_vacia_y_absoluta(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    vacia = read_text_file(path="   ")
    absoluta = read_text_file(path=str(tmp_path / "x.txt"))

    assert "vacía" in vacia and "relativa" in vacia
    assert "absoluta" in absoluta


def test_lector_rechaza_escapar_del_sandbox(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    out = read_text_file(path="../secreto.txt")
    assert out.startswith("Error:") and ".." in out


def test_lector_archivo_inexistente_lista_el_directorio(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "datos").mkdir()
    (tmp_path / "datos" / "notas.txt").write_text("hola", encoding="utf-8")

    out = read_text_file(path="datos/nota.txt")

    assert out.startswith("Error:")
    assert "notas.txt" in out, "debe listar los archivos disponibles"


def test_lector_no_lista_entradas_ocultas(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("SECRETO=1", encoding="utf-8")
    (tmp_path / "datos.txt").write_text("hola", encoding="utf-8")

    out = read_text_file(path="datos.csv")

    assert "datos.txt" in out
    assert ".env" not in out


def test_lector_directorio_lista_contenido(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "datos").mkdir()
    (tmp_path / "datos" / "a.txt").write_text("a", encoding="utf-8")

    out = read_text_file(path="datos")

    assert "es un directorio" in out
    assert "a.txt" in out


# --------------------------------------------------------------------------
# Recuperación de punta a punta: el LLM corrige a partir del mensaje de error.
# --------------------------------------------------------------------------


def test_recuperacion_calculadora_en_el_bucle() -> None:
    """Primer intento con operador inválido; el agente le devuelve la lista
    de operadores y el segundo intento sale bien."""
    mock = MockLLMClient(
        [
            LLMResponse(
                content=None,
                tool_calls=[
                    _tc(
                        "calculator",
                        {"left_operand": 10, "right_operand": 3, "operator": "^"},
                        "c1",
                    )
                ],
            ),
            LLMResponse(
                content=None,
                tool_calls=[
                    _tc(
                        "calculator",
                        {"left_operand": 10, "right_operand": 3, "operator": "%"},
                        "c2",
                    )
                ],
            ),
            LLMResponse(content="El resto es 1."),
        ]
    )
    agent = _agente(mock)
    result = agent.run("calculá 10 elevado a 3, o el resto si no podés")

    assert result.answer == "El resto es 1."
    assert result.steps[0].tool_output.startswith("Error:")
    assert result.steps[1].tool_output == "1"


def test_recuperacion_lector_en_el_bucle(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "informe.txt").write_text("contenido real", encoding="utf-8")

    mock = MockLLMClient(
        [
            LLMResponse(
                content=None,
                tool_calls=[_tc("read_text_file", {"path": "informes.txt"}, "c1")],
            ),
            LLMResponse(
                content=None,
                tool_calls=[_tc("read_text_file", {"path": "informe.txt"}, "c2")],
            ),
            LLMResponse(content="El archivo dice: contenido real"),
        ]
    )
    agent = _agente(mock)
    result = agent.run("leé el informe")

    assert "informe.txt" in result.steps[0].tool_output
    assert result.steps[1].tool_output == "contenido real"
