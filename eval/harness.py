"""Ejecución de una corrida y captura de su traza."""

from __future__ import annotations

import copy
import os
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from mia_agents.llm_client import LLMClient
from mia_world import Scenario, check_goal, make_world_tools

from eval.configs import ExperimentConfig, OPTIMAL_CALLS, estimate_cost


@dataclass
class ToolCallRecord:
    index: int
    name: str | None
    arguments: str | None
    output: str | None
    is_error: bool
    framework_error: str | None = None


@dataclass
class RunRecord:
    """Todo lo observable de una corrida. Se serializa a `runs.jsonl`."""

    run_id: str
    timestamp: float
    scenario_id: str
    difficulty: str
    experiment: str
    rep: int
    model_id: str

    goal_achieved: bool
    goal_reason: str

    optimal_calls: int
    step_budget: int
    n_calls: int
    n_error_calls: int

    answer: str
    agent_error: str | None
    exhausted_steps: bool

    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float
    wall_seconds: float

    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    final_room: str = ""
    final_inventory: list[str] = field(default_factory=list)
    event_log: list[str] = field(default_factory=list)
    crashed: str | None = None

    @property
    def efficiency(self) -> float | None:
        """Óptimo sobre llamadas usadas. Indefinida si la corrida no ganó."""
        if not self.goal_achieved or self.n_calls == 0:
            return None
        return self.optimal_calls / self.n_calls

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["efficiency"] = self.efficiency
        return data


def _model_id() -> str:
    return os.environ.get("BEDROCK_MODEL_ID", "desconocido")


def _build_trace(steps: list[Any]) -> tuple[list[ToolCallRecord], int]:
    """Traza a partir de los `AgentStep`.

    Se leen del `AgentResult` y no de los callables porque los steps
    también registran lo que nunca llegó al mundo: herramientas
    alucinadas, JSON malformado y argumentos que no encajan en la firma.
    """
    trace: list[ToolCallRecord] = []
    n_errors = 0
    for i, step in enumerate(steps):
        output = step.tool_output or ""
        is_error = bool(step.error) or output.startswith("Error:")
        if is_error:
            n_errors += 1
        trace.append(
            ToolCallRecord(
                index=i,
                name=step.tool_name,
                arguments=step.tool_input,
                output=step.tool_output,
                is_error=is_error,
                framework_error=step.error,
            )
        )
    return trace, n_errors


def run_once(
    scenario: Scenario,
    experiment: ExperimentConfig,
    rep: int,
    *,
    llm_client: LLMClient | None = None,
    build_agent: Any = None,
) -> RunRecord:
    """Corre el agente una vez sobre un escenario.

    `llm_client` y `build_agent` se inyectan desde los tests para
    ejercitar el harness con un mock.
    """
    if build_agent is None:
        from student_framework import build_agent as _build_agent

        build_agent = _build_agent

    # Las herramientas mutan el World en sitio: sin copia, la repetición 2
    # heredaría la puerta que abrió la 1.
    world = copy.deepcopy(scenario.initial_world)
    tools = make_world_tools(world)
    if experiment.transform_tools is not None:
        tools = experiment.transform_tools(tools)

    config: dict[str, Any] = dict(experiment.agent_config(scenario.id))
    if llm_client is not None:
        config["llm_client"] = llm_client

    started = time.monotonic()
    crashed: str | None = None
    answer = ""
    agent_error: str | None = None
    trace: list[ToolCallRecord] = []
    n_errors = 0
    input_tokens: int | None = None
    output_tokens: int | None = None

    try:
        agent = build_agent(config)
        for fn, schema in tools:
            agent.register_tool(fn, schema)
        result = agent.run(scenario.user_message)
        answer = result.answer
        agent_error = result.error
        input_tokens = result.input_tokens
        output_tokens = result.output_tokens
        trace, n_errors = _build_trace(result.steps)
    except Exception:  # noqa: BLE001
        # Una corrida que revienta es un dato del experimento, no un motivo
        # para abortar el barrido.
        crashed = traceback.format_exc(limit=6)

    wall = time.monotonic() - started
    achieved, reason = check_goal(world, scenario.goal)

    return RunRecord(
        run_id=uuid.uuid4().hex[:12],
        timestamp=time.time(),
        scenario_id=scenario.id,
        difficulty=scenario.difficulty,
        experiment=experiment.name,
        rep=rep,
        model_id=_model_id(),
        goal_achieved=achieved,
        goal_reason=reason,
        optimal_calls=OPTIMAL_CALLS.get(scenario.id, 0),
        step_budget=experiment.step_budget(scenario.id),
        n_calls=len(trace),
        n_error_calls=n_errors,
        answer=answer,
        agent_error=agent_error,
        exhausted_steps=bool(agent_error and "iteraciones" in agent_error),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=estimate_cost(_model_id(), input_tokens or 0, output_tokens or 0),
        wall_seconds=round(wall, 2),
        tool_calls=trace,
        final_room=world.current_room,
        final_inventory=list(world.inventory),
        event_log=list(world.event_log),
        crashed=crashed,
    )


def preflight(model_id: str | None = None) -> tuple[bool, str]:
    """Valida el proveedor con una llamada mínima antes del barrido."""
    try:
        client = LLMClient.from_env()
        resp = client.chat(messages=[{"role": "user", "content": "Respondé: OK"}])
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    return True, (resp.content or "").strip()[:80]
