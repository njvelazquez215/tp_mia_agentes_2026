"""Escenarios de prueba propios del grupo (M1).

Estos tests NO son los de conformidad (esos viven en
`tests/conformance/test_m1.py` y no se editan). Acá demostramos el
comportamiento de nuestras tres herramientas y, sobre todo, escenarios
donde el agente usa **al menos dos herramientas** en una misma corrida,
tal como pide el README de M1.

Usan el `MockLLMClient` provisto: programamos las respuestas del LLM
(tool_calls + respuesta final) para guiar el bucle de forma determinista,
sin consumir créditos de API.
"""

from __future__ import annotations

import json

from mia_agents.testing import MockLLMClient
from mia_agents.types import AgentStep, LLMResponse, ToolCall

from student_framework import build_agent
from student_framework.tools.calculator import calculator
from student_framework.tools.file_reader import read_text_file
from student_framework.tools.word_counter import count_words


def _tc(name: str, args: dict, call_id: str = "c1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=json.dumps(args))


# --------------------------------------------------------------------------
# Tests unitarios de cada herramienta (lógica pura, sin agente).
# --------------------------------------------------------------------------


def test_calculator_operadores() -> None:
    assert calculator(left_operand=17, right_operand=23, operator="*") == "391"
    assert calculator(left_operand=2, right_operand=2, operator="+") == "4"
    assert calculator(left_operand=10, right_operand=3, operator="%") == "1"
    assert calculator(left_operand=9, right_operand=2, operator="/") == "4.5"


def test_calculator_division_por_cero() -> None:
    out = calculator(left_operand=1, right_operand=0, operator="/")
    assert "cero" in out.lower()


def test_calculator_operador_invalido() -> None:
    out = calculator(left_operand=1, right_operand=2, operator="^")
    assert "no soportado" in out.lower()


def test_file_reader_archivo_inexistente() -> None:
    out = read_text_file(path="no_existe_xyz.txt")
    assert out.startswith("Error:")


def test_file_reader_lee_contenido(tmp_path) -> None:
    f = tmp_path / "hola.txt"
    f.write_text("línea uno\nlínea dos\n", encoding="utf-8")
    assert read_text_file(path=str(f)) == "línea uno\nlínea dos\n"


def test_word_counter() -> None:
    out = count_words(text="hola mundo cruel")
    assert "palabras: 3" in out


# --------------------------------------------------------------------------
# Escenario 1 (multi-tool): leer un archivo y CONTAR sus palabras.
# El agente usa file_reader y, con su salida, word_counter.
# --------------------------------------------------------------------------


def test_escenario_leer_y_contar(tmp_path) -> None:
    archivo = tmp_path / "poema.txt"
    archivo.write_text("rosas son rojas violetas azules", encoding="utf-8")

    mock = MockLLMClient(
        [
            # Turno 1: el LLM pide leer el archivo.
            LLMResponse(
                content=None,
                tool_calls=[_tc("read_text_file", {"path": str(archivo)}, "a1")],
            ),
            # Turno 2: con el contenido, pide contar palabras.
            LLMResponse(
                content=None,
                tool_calls=[
                    _tc(
                        "count_words",
                        {"text": "rosas son rojas violetas azules"},
                        "a2",
                    )
                ],
            ),
            # Turno 3: respuesta final de texto (sin tool_calls).
            LLMResponse(content="El archivo tiene 5 palabras."),
        ]
    )
    agent = build_agent({"llm_client": mock})
    result = agent.run("¿Cuántas palabras tiene poema.txt?")

    assert result.answer == "El archivo tiene 5 palabras."
    assert [s.tool_name for s in result.steps] == ["read_text_file", "count_words"]
    assert all(s.error is None for s in result.steps)
    # Se llamó al LLM 3 veces (2 con tool_calls + 1 respuesta final).
    assert mock.call_count == 3
    # El resultado de leer el archivo se volcó como contexto al LLM.
    segunda_llamada = mock.calls[1]["messages"]
    assert any(m.get("role") == "tool" for m in segunda_llamada)


# --------------------------------------------------------------------------
# Escenario 2 (multi-tool): dos operaciones con la calculadora encadenadas.
# --------------------------------------------------------------------------


def test_escenario_calculo_encadenado() -> None:
    mock = MockLLMClient(
        [
            LLMResponse(
                content=None,
                tool_calls=[
                    _tc("calculator", {"left_operand": 17, "right_operand": 23,
                                        "operator": "*"}, "a1")
                ],
            ),
            LLMResponse(
                content=None,
                tool_calls=[
                    _tc("calculator", {"left_operand": 391, "right_operand": 9,
                                        "operator": "+"}, "a2")
                ],
            ),
            LLMResponse(content="17 * 23 + 9 = 400."),
        ]
    )
    agent = build_agent({"llm_client": mock})
    result = agent.run("Calculá 17 * 23 y sumale 9.")

    assert result.answer == "17 * 23 + 9 = 400."
    assert len(result.steps) == 2
    assert result.steps[0].tool_output == "391"
    assert result.steps[1].tool_output == "400"


# --------------------------------------------------------------------------
# Robustez: herramienta desconocida (alucinada) -> AgentStep con error.
# --------------------------------------------------------------------------


def test_herramienta_desconocida_no_rompe() -> None:
    mock = MockLLMClient(
        [
            LLMResponse(
                content=None,
                tool_calls=[_tc("teletransportar", {"destino": "Marte"}, "a1")],
            ),
            LLMResponse(content="No tengo esa herramienta."),
        ]
    )
    agent = build_agent({"llm_client": mock})
    result = agent.run("Teletransportame a Marte.")

    assert result.answer == "No tengo esa herramienta."
    assert len(result.steps) == 1
    assert result.steps[0].error is not None
    assert result.steps[0].tool_output is None


# --------------------------------------------------------------------------
# Robustez: bucle infinito de tool_calls -> corta por max_iterations.
# --------------------------------------------------------------------------


def test_corta_por_max_iterations() -> None:
    # El LLM "se obsesiona" y siempre pide la calculadora, nunca cierra.
    loop_response = LLMResponse(
        content=None,
        tool_calls=[_tc("calculator", {"left_operand": 1, "right_operand": 1,
                                        "operator": "+"})],
    )
    mock = MockLLMClient([loop_response] * 50)  # de sobra
    agent = build_agent({"llm_client": mock})
    result = agent.run("entrá en loop")

    # Devuelve un AgentResult válido aunque haya cortado por límite.
    assert result.error is not None
    # No llamó al LLM más que max_iterations veces (default 10).
    assert mock.call_count == 10


# --------------------------------------------------------------------------
# Entradas básicas: cadena vacía no debe romper.
# --------------------------------------------------------------------------


def test_cadena_vacia() -> None:
    mock = MockLLMClient([LLMResponse(content="¿En qué puedo ayudarte?")])
    agent = build_agent({"llm_client": mock})
    result = agent.run("")
    assert result.answer == "¿En qué puedo ayudarte?"
    assert result.steps == []
