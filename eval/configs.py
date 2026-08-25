"""Escenarios, tarifas, prompts y definición de los experimentos."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from mia_agents.types import ToolSchema

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENARIOS_DIR = REPO_ROOT / "scenarios"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# Escenarios
# ---------------------------------------------------------------------------

# Longitud de la solución óptima. Coincide con las secuencias de
# `test_m3_world.py::_SCENARIO_SOLUTIONS`. Denominador de la eficiencia.
OPTIMAL_CALLS: dict[str, int] = {
    "study-with-key": 3,
    "color-locks": 11,
    "apartment-keys": 7,
    "library-search": 7,
    "office-sequence": 13,
    "extreme-archive": 4,
    "vault-combination": 21,
    "backtracking-vault": 18,
}

# Columna "brute-force peor caso" del enunciado: lo que cuesta resolver el
# escenario sin saber de antemano dónde está el objeto clave. El enunciado
# no publica el de `extreme-archive`; son 23 (estantería + 20 expedientes
# + take + use).
BRUTE_FORCE_CALLS: dict[str, int] = {
    "study-with-key": 3,
    "color-locks": 11,
    "apartment-keys": 7,
    "library-search": 13,
    "office-sequence": 13,
    "extreme-archive": 23,
    "vault-combination": 21,
    "backtracking-vault": 18,
}

# Margen para tomar y usar el objeto una vez hallado.
_HOLGURA = 4


def budget_generoso(scenario_id: str) -> int:
    """Presupuesto que alcanza para buscar y además resolver."""
    optimo = OPTIMAL_CALLS.get(scenario_id, 10)
    bruto = BRUTE_FORCE_CALLS.get(scenario_id, optimo)
    return max(2 * optimo, bruto + _HOLGURA)


DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2, "extreme": 3}


# ---------------------------------------------------------------------------
# Tarifas (USD por millón de tokens, Bedrock on-demand)
# ---------------------------------------------------------------------------

PRICING: dict[str, tuple[float, float]] = {
    "amazon.nova-micro-v1:0": (0.035, 0.14),
    "amazon.nova-lite-v1:0": (0.06, 0.24),
    "amazon.nova-pro-v1:0": (0.80, 3.20),
}


def estimate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Coste en USD. 0.0 si el modelo no está tarifado."""
    price_in, price_out = PRICING.get(model_id, (0.0, 0.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

STRATEGIC_PROMPT = """\
Sos un agente que resuelve salas de escape en un mundo simulado. Tu objetivo \
es cumplir la meta que te plantea el usuario actuando sobre el mundo con las \
herramientas disponibles.

REGLAS DEL MUNDO
- Las herramientas reciben SIEMPRE el id del objeto, no su nombre en prosa. \
Los ids aparecen entre corchetes en las descripciones: en \
"llave dorada [id: llave_oro]" el argumento correcto es llave_oro.
- look describe la sala: objetos visibles, estado de las puertas, salidas e \
inventario. Usalo cuando no sepas dónde estás o qué hay.
- examine sobre un contenedor revela lo que hay dentro. Los objetos ocultos NO \
son visibles ni tomables hasta que examinás el contenedor que los esconde.
- take mueve un objeto visible a tu inventario. Solo podés usar (use) objetos \
que ya estén en tu inventario.
- use aplica un objeto del inventario sobre otro de la sala. Es la única forma \
de abrir cerraduras.
- Si tenés la herramienta go, el mundo tiene varias salas. Una salida puede \
estar bloqueada por una puerta que primero hay que abrir.

CÓMO TRABAJAR
1. Empezá con look para orientarte.
2. Antes de actuar, decidí qué te falta para cumplir la meta y cuál es el \
próximo subobjetivo. Si la meta pide varias cosas en cierto orden, respetá ese \
orden: algunas acciones son irreversibles.
3. Llevá la cuenta de qué salas visitaste, qué contienen y qué tenés en el \
inventario. No repitas una acción que ya hiciste.
4. Si una herramienta devuelve un texto que empieza con "Error:", leelo: dice \
exactamente qué falló. Corregí el argumento en vez de reintentar igual.
5. Cuando la meta esté cumplida, respondé con texto y sin llamar herramientas, \
explicando brevemente qué hiciste.

Actuá. No pidas permiso ni describas lo que vas a hacer sin hacerlo.\
"""

MINIMAL_PROMPT = """\
Sos un agente que resuelve salas de escape en un mundo simulado. Usá las \
herramientas disponibles para cumplir la meta que te plantea el usuario. Las \
herramientas reciben el id del objeto, que aparece entre corchetes como \
"[id: ...]". Cuando termines, respondé con texto.\
"""


# ---------------------------------------------------------------------------
# Experimentos
# ---------------------------------------------------------------------------

ToolPair = tuple[Callable[..., str], ToolSchema]


def _noop_look(tools: list[ToolPair]) -> list[ToolPair]:
    """Deja `look` anunciado al modelo pero sin información en la respuesta."""
    out: list[ToolPair] = []
    for fn, schema in tools:
        if schema.name == "look":
            out.append((lambda: "No observás nada nuevo.", schema))
        else:
            out.append((fn, schema))
    return out


@dataclass(frozen=True)
class ExperimentConfig:
    """Una celda del banco de pruebas.

    `budget_basis` elige contra qué se mide el tope de pasos: "generoso"
    usa `budget_generoso`, "optimo" la solución perfecta. El presupuesto
    se resuelve por escenario; un tope fijo favorecería a los cortos.
    """

    name: str
    description: str
    system_prompt: str = STRATEGIC_PROMPT
    budget_basis: str = "generoso"
    step_multiplier: float = 1.0
    max_history_messages: int = 50
    transform_tools: Callable[[list[ToolPair]], list[ToolPair]] | None = None
    extra_config: dict[str, Any] = field(default_factory=dict)

    def step_budget(self, scenario_id: str) -> int:
        if self.budget_basis == "optimo":
            base = OPTIMAL_CALLS.get(scenario_id, 10)
        else:
            base = budget_generoso(scenario_id)
        return max(3, int(round(base * self.step_multiplier)))

    def agent_config(self, scenario_id: str) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "system_prompt": self.system_prompt,
            "max_iterations": self.step_budget(scenario_id),
            "max_history_messages": self.max_history_messages,
        }
        cfg.update(self.extra_config)
        return cfg


BASELINE = ExperimentConfig(
    name="baseline",
    description=(
        "Referencia: prompt con estrategia explícita, presupuesto generoso, "
        "ventana de historial de 50 mensajes y herramientas intactas."
    ),
)

EXPERIMENTS: dict[str, ExperimentConfig] = {
    "baseline": BASELINE,
    "steps-justo": ExperimentConfig(
        name="steps-justo",
        description=(
            "Presupuesto igual a la solución óptima: cero margen de error y, "
            "en los escenarios de búsqueda, ni siquiera lo justo para buscar."
        ),
        budget_basis="optimo",
        step_multiplier=1.0,
    ),
    "steps-holgado": ExperimentConfig(
        name="steps-holgado",
        description=(
            "El doble del presupuesto generoso. Si el goal rate no sube "
            "respecto del baseline, el cuello de botella no son los pasos."
        ),
        step_multiplier=2.0,
    ),
    "memoria-corta": ExperimentConfig(
        name="memoria-corta",
        description=(
            "Ventana de historial recortada a 12 mensajes. Mide cuánto aporta "
            "la estrategia de memoria del M2."
        ),
        max_history_messages=12,
    ),
    "prompt-minimo": ExperimentConfig(
        name="prompt-minimo",
        description=(
            "System prompt sin procedimiento: qué herramientas hay, pero no "
            "cómo encarar el problema."
        ),
        system_prompt=MINIMAL_PROMPT,
    ),
    "look-noop": ExperimentConfig(
        name="look-noop",
        description=(
            "look devuelve texto vacío. Mide la dependencia de reobservar el "
            "mundo frente a recordarlo."
        ),
        transform_tools=_noop_look,
    ),
}

DEFAULT_EXPERIMENTS = [
    "baseline",
    "steps-justo",
    "steps-holgado",
    "memoria-corta",
    "prompt-minimo",
    "look-noop",
]
