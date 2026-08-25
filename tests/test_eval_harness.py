"""Tests de la infraestructura de evaluación de M3.

Usan `MockLLMClient`, así que son deterministas y no consumen API.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mia_agents.testing.mock_llm import MockLLMClient
from mia_agents.types import LLMResponse, ToolCall
from mia_world import load_scenario

from eval.configs import EXPERIMENTS, ExperimentConfig, OPTIMAL_CALLS, estimate_cost
from eval.harness import run_once
from eval.metrics import (
    afirma_exito_falso,
    agregar,
    clasificar_friccion,
    clasificar_terminacion,
)
from eval.report import construir_informe

SCENARIOS = Path(__file__).resolve().parents[1] / "scenarios"

SOLUCION_ESTUDIO = [
    ("examine", {"target": "alfombra"}),
    ("take", {"item": "llave_oro"}),
    ("use", {"item": "llave_oro", "target": "puerta_principal"}),
]


def _guion(acciones: list[tuple[str, dict]], cierre: str) -> MockLLMClient:
    """Mock que emite una tool call por vuelta y cierra con texto."""
    respuestas: list[LLMResponse] = [
        LLMResponse(
            content=None,
            tool_calls=[ToolCall(f"c{i}", nombre, json.dumps(kwargs))],
            input_tokens=100,
            output_tokens=20,
        )
        for i, (nombre, kwargs) in enumerate(acciones)
    ]
    respuestas.append(LLMResponse(content=cierre, input_tokens=100, output_tokens=20))
    return MockLLMClient(respuestas)


def _correr(scenario_file: str, acciones, cierre="Listo.", exp=None):
    from student_framework import build_agent

    sc = load_scenario(SCENARIOS / scenario_file)
    return run_once(
        sc,
        exp or EXPERIMENTS["baseline"],
        rep=0,
        llm_client=_guion(acciones, cierre),
        build_agent=build_agent,
    )


# --- El veredicto sale del mundo, no del texto del agente ------------------


def test_solucion_optima_se_registra_como_exito() -> None:
    rec = _correr("01-study-with-key.json", SOLUCION_ESTUDIO)
    assert rec.goal_achieved is True
    assert rec.n_calls == 3
    assert rec.n_error_calls == 0
    assert rec.efficiency == pytest.approx(1.0)
    assert rec.final_inventory == ["llave_oro"]


def test_agente_que_miente_no_cuenta_como_exito() -> None:
    rec = _correr(
        "01-study-with-key.json",
        [("look", {})],
        cierre="Ya abrí la puerta principal, estás libre.",
    )
    assert rec.goal_achieved is False
    assert afirma_exito_falso(rec.to_json()) is True
    assert clasificar_terminacion(rec.to_json()) == "rendicion_temprana"


def test_eficiencia_indefinida_si_pierde() -> None:
    rec = _correr("01-study-with-key.json", [("look", {})])
    assert rec.goal_achieved is False
    assert rec.efficiency is None


def test_ruta_suboptima_gana_pero_baja_la_eficiencia() -> None:
    rec = _correr("01-study-with-key.json", [("look", {}), ("look", {})] + SOLUCION_ESTUDIO)
    assert rec.goal_achieved is True
    assert rec.n_calls == 5
    assert rec.efficiency == pytest.approx(3 / 5)


# --- Aislamiento entre corridas -------------------------------------------


def test_corridas_sucesivas_no_comparten_mundo() -> None:
    """Sin la copia profunda, la segunda corrida heredaría la puerta abierta."""
    from student_framework import build_agent

    sc = load_scenario(SCENARIOS / "01-study-with-key.json")
    ganada = run_once(
        sc, EXPERIMENTS["baseline"], 0,
        llm_client=_guion(SOLUCION_ESTUDIO, "Listo."), build_agent=build_agent,
    )
    perdida = run_once(
        sc, EXPERIMENTS["baseline"], 1,
        llm_client=_guion([("look", {})], "No pude."), build_agent=build_agent,
    )
    assert ganada.goal_achieved is True
    assert perdida.goal_achieved is False, "el mundo se filtró entre corridas"


# --- Presupuesto de pasos --------------------------------------------------


def test_presupuesto_se_escala_por_escenario() -> None:
    base = EXPERIMENTS["baseline"]
    assert base.step_budget("vault-combination") == 42
    assert base.step_budget("study-with-key") == 7
    assert EXPERIMENTS["steps-justo"].step_budget("color-locks") == 11
    assert EXPERIMENTS["steps-holgado"].step_budget("color-locks") == 44


def test_escenarios_de_busqueda_reciben_presupuesto_para_buscar() -> None:
    """El óptimo presupone saber dónde mirar; buscar cuesta aparte.

    Con el tope en 2x el óptimo, el agente se queda sin pasos mientras
    todavía busca y el fallo mide el presupuesto, no su razonamiento.
    """
    base = EXPERIMENTS["baseline"]
    assert base.step_budget("library-search") == 17
    assert base.step_budget("extreme-archive") == 27
    for esc in ("library-search", "extreme-archive"):
        assert base.step_budget(esc) > 2 * OPTIMAL_CALLS[esc]


def test_presupuesto_agotado_se_clasifica_aparte() -> None:
    exp = ExperimentConfig(name="corto", description="", budget_basis="optimo")
    rec = _correr("01-study-with-key.json", [("look", {})] * 10, exp=exp)
    assert rec.goal_achieved is False
    assert rec.exhausted_steps is True
    assert clasificar_terminacion(rec.to_json()) == "presupuesto_agotado"


# --- Taxonomía de fricción -------------------------------------------------


@pytest.mark.parametrize(
    "acciones,esperado",
    [
        ([("take", {"item": "llave dorada"})], "id_inexistente"),
        ([("take", {"item": "llave_oro"})], "objeto_no_visible"),
        ([("use", {"item": "llave_oro", "target": "puerta_principal"})], "inventario_vacio"),
        ([("examine", {"target": "alfombra"})] * 4, "repeticion"),
    ],
)
def test_friccion_dominante(acciones, esperado) -> None:
    rec = _correr("01-study-with-key.json", acciones)
    assert clasificar_friccion(rec.to_json()) == esperado


def test_herramienta_alucinada_es_error_de_argumentos() -> None:
    rec = _correr("01-study-with-key.json", [("teletransportar", {"a": "afuera"})])
    assert rec.n_error_calls == 1
    assert clasificar_friccion(rec.to_json()) == "argumentos_invalidos"


def test_navegacion_invalida_en_multi_sala() -> None:
    rec = _correr("05-medium-apartment-keys.json", [("go", {"direction": "arriba"})])
    assert clasificar_friccion(rec.to_json()) == "navegacion_invalida"


# --- Ablación de herramienta ----------------------------------------------


def test_look_noop_no_revela_el_mundo() -> None:
    rec = _correr("01-study-with-key.json", [("look", {})], exp=EXPERIMENTS["look-noop"])
    assert rec.n_calls == 1
    assert "alfombra" not in (rec.tool_calls[0].output or "")

    normal = _correr("01-study-with-key.json", [("look", {})])
    assert "alfombra" in (normal.tool_calls[0].output or "")


# --- Robustez del harness --------------------------------------------------


def test_excepcion_del_agente_no_aborta_el_barrido() -> None:
    from student_framework import build_agent

    sc = load_scenario(SCENARIOS / "01-study-with-key.json")
    roto = MockLLMClient([ValueError("modelo inexistente")])
    rec = run_once(sc, EXPERIMENTS["baseline"], 0, llm_client=roto, build_agent=build_agent)
    assert rec.crashed is not None
    assert rec.goal_achieved is False
    assert clasificar_terminacion(rec.to_json()) == "crash"


def test_optimal_calls_cubre_todos_los_escenarios() -> None:
    """Si falta un escenario, su eficiencia se calcularía contra 0."""
    from mia_world import list_scenarios

    for sc in list_scenarios(SCENARIOS):
        assert sc.id in OPTIMAL_CALLS, f"falta el óptimo de {sc.id}"
        assert OPTIMAL_CALLS[sc.id] > 0


# --- Agregación e informe --------------------------------------------------


def test_pass_at_k_distingue_varianza_de_incapacidad() -> None:
    runs = [
        {"scenario_id": "a", "goal_achieved": True, "n_calls": 3, "n_error_calls": 0,
         "efficiency": 1.0, "tool_calls": [], "answer": "", "difficulty": "easy"},
        {"scenario_id": "a", "goal_achieved": False, "n_calls": 3, "n_error_calls": 0,
         "efficiency": None, "tool_calls": [], "answer": "", "difficulty": "easy"},
    ]
    a = agregar(runs, "x")
    assert a.goal_rate == pytest.approx(0.5)
    assert a.pass_at_k == pytest.approx(1.0)
    assert a.k == 2


def test_informe_se_genera_con_datos_reales() -> None:
    ganada = _correr("01-study-with-key.json", SOLUCION_ESTUDIO).to_json()
    perdida = _correr("01-study-with-key.json", [("look", {})]).to_json()
    md = construir_informe([ganada, perdida])
    assert "# Resultados de la evaluación" in md
    assert "study-with-key" in md
    assert "50%" in md


def test_informe_no_revienta_sin_corridas() -> None:
    assert "No hay corridas" in construir_informe([])


def test_coste_usa_la_tarifa_del_modelo() -> None:
    assert estimate_cost("amazon.nova-lite-v1:0", 1_000_000, 0) == pytest.approx(0.06)
    assert estimate_cost("amazon.nova-lite-v1:0", 0, 1_000_000) == pytest.approx(0.24)
    assert estimate_cost("modelo-raro", 1_000_000, 1_000_000) == 0.0


# --- Configuración del juez ------------------------------------------------


def test_modelo_juez_usa_inference_profile() -> None:
    """Nova Pro rechaza on-demand con el id pelado; necesita el perfil `us.`."""
    from eval.judge import JUDGE_MODEL_DEFAULT

    assert JUDGE_MODEL_DEFAULT.startswith("us."), JUDGE_MODEL_DEFAULT


def test_juez_es_distinto_del_actor() -> None:
    """Un modelo que se evalúa a sí mismo tiende a preferir su propia salida."""
    import os

    from eval.judge import modelo_juez

    actor = os.environ.get("BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0")
    assert modelo_juez() != actor
