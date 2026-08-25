"""Generación de `results/report.md` a partir de los JSONL."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eval.configs import DIFFICULTY_ORDER, EXPERIMENTS, OPTIMAL_CALLS
from eval.judge import DIMENSIONES, promedios_rubrica
from eval.metrics import (
    FRICCION,
    TERMINACION,
    Agregado,
    afirma_exito_falso,
    agregar,
    agrupar,
    clasificar_friccion,
    clasificar_terminacion,
)


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _opt(x: float | None, fmt: str = "{:.2f}") -> str:
    return "—" if x is None else fmt.format(x)


def _tabla(cabeceras: list[str], filas: list[list[str]]) -> str:
    if not filas:
        return "_(sin datos)_\n"
    lineas = [
        "| " + " | ".join(cabeceras) + " |",
        "| " + " | ".join(["---"] * len(cabeceras)) + " |",
    ]
    for f in filas:
        lineas.append("| " + " | ".join(f) + " |")
    return "\n".join(lineas) + "\n"


_CABECERA_AGREGADO = [
    "",
    "n",
    "goal rate",
    "pass@k",
    "eficiencia",
    "calls",
    "err/call",
    "coste",
]


def _fila_agregado(nombre: str, a: Agregado) -> list[str]:
    return [
        nombre,
        str(a.n_runs),
        f"{_pct(a.goal_rate)} ± {_pct(a.goal_rate_std)}",
        _pct(a.pass_at_k),
        _opt(a.eficiencia_media),
        f"{a.calls_media:.1f}",
        _pct(a.tasa_error_calls),
        f"${a.coste_usd:.4f}",
    ]


def construir_informe(
    runs: list[dict[str, Any]],
    veredictos: dict[str, dict[str, Any]] | None = None,
) -> str:
    veredictos = veredictos or {}
    partes: list[str] = []
    add = partes.append

    add("# Resultados de la evaluación — Milestone 3\n")
    if not runs:
        add("_No hay corridas registradas._\n")
        return "\n".join(partes)

    modelo = runs[0].get("model_id", "desconocido")
    n_exp = len(agrupar(runs, "experiment"))
    n_esc = len(agrupar(runs, "scenario_id"))
    coste = sum(r.get("cost_usd") or 0.0 for r in runs)
    segundos = sum(r.get("wall_seconds") or 0.0 for r in runs)

    add(
        f"Modelo del agente: `{modelo}`. "
        f"{len(runs)} corridas = {n_esc} escenarios x {n_exp} configuraciones x "
        f"repeticiones. Coste total estimado: **${coste:.4f}**. "
        f"Tiempo de cómputo: {segundos / 60:.1f} min.\n"
    )

    base = [r for r in runs if r["experiment"] == "baseline"]

    add("\n## 1. Resultado principal (baseline)\n")
    if base:
        a = agregar(base, "baseline")
        add(
            f"El agente resuelve **{_pct(a.goal_rate)}** de las corridas "
            f"(pass@{a.k} = {_pct(a.pass_at_k)} de los escenarios). Cuando gana "
            f"alcanza {_opt(a.eficiencia_media)} de eficiencia respecto de la "
            f"solución óptima, y **{_pct(a.tasa_error_calls)}** de sus llamadas "
            f"a herramienta devuelven error.\n"
        )
        add("\n" + _tabla(_CABECERA_AGREGADO, [_fila_agregado("baseline", a)]))
    else:
        add("_(no se corrió el baseline)_\n")

    add("\n## 2. Desglose por escenario (baseline)\n")
    add(
        "`óptimo` es la longitud de la solución perfecta; `calls`, lo que gastó "
        "el agente. La eficiencia solo se define en corridas ganadas.\n\n"
    )
    filas = []
    for esc, rs in sorted(
        agrupar(base, "scenario_id").items(),
        key=lambda kv: (
            DIFFICULTY_ORDER.get(kv[1][0]["difficulty"], 99),
            OPTIMAL_CALLS.get(kv[0], 0),
        ),
    ):
        a = agregar(rs, esc)
        filas.append(
            [
                esc,
                rs[0]["difficulty"],
                str(OPTIMAL_CALLS.get(esc, "?")),
                str(rs[0]["step_budget"]),
                f"{a.n_exitos}/{a.n_runs}",
                f"{a.calls_media:.1f}",
                _opt(a.eficiencia_media),
                _pct(a.tasa_error_calls),
            ]
        )
    add(
        _tabla(
            ["escenario", "dif.", "óptimo", "tope", "éxitos", "calls", "efic.", "err/call"],
            filas,
        )
    )

    add("\n## 3. Desglose por dificultad (baseline)\n")
    filas = [
        _fila_agregado(dif, agregar(rs, dif))
        for dif, rs in sorted(
            agrupar(base, "difficulty").items(),
            key=lambda kv: DIFFICULTY_ORDER.get(kv[0], 99),
        )
    ]
    add(_tabla(_CABECERA_AGREGADO, filas))

    add("\n## 4. Análisis de errores\n")
    add(
        "Dos ejes independientes. **Terminación** es por qué se detuvo el "
        "agente; **fricción**, qué tipo de error predominó en su traza. Una "
        "corrida puede agotar el presupuesto y además haber estado peleando "
        "con ids inexistentes: son dos hechos distintos y cada uno sugiere un "
        "arreglo distinto.\n"
    )

    add("\n### 4.1 Cómo terminaron las corridas (todas las configuraciones)\n\n")
    conteo_t: dict[str, int] = {}
    for r in runs:
        c = clasificar_terminacion(r)
        conteo_t[c] = conteo_t.get(c, 0) + 1
    add(
        _tabla(
            ["terminación", "n", "%", "qué significa"],
            [
                [cat, str(n), _pct(n / len(runs)), TERMINACION.get(cat, "")]
                for cat, n in sorted(conteo_t.items(), key=lambda kv: -kv[1])
            ],
        )
    )

    fallidas = [r for r in runs if not r["goal_achieved"]]
    add("\n### 4.2 Fricción dominante en las corridas fallidas\n\n")
    if fallidas:
        conteo_f: dict[str, int] = {}
        for r in fallidas:
            c = clasificar_friccion(r)
            conteo_f[c] = conteo_f.get(c, 0) + 1
        add(
            _tabla(
                ["fricción", "n", "%", "qué significa"],
                [
                    [cat, str(n), _pct(n / len(fallidas)), FRICCION.get(cat, "")]
                    for cat, n in sorted(conteo_f.items(), key=lambda kv: -kv[1])
                ],
            )
        )
    else:
        add("_No hubo corridas fallidas._\n")

    falsas = sum(1 for r in fallidas if afirma_exito_falso(r))
    add(
        f"\n**Éxito alucinado:** en {falsas} de {len(fallidas)} corridas "
        f"fallidas el agente afirmó haber abierto la puerta mientras el mundo "
        f"decía lo contrario. Es la razón por la que la métrica se calcula con "
        f"`check_goal` sobre el estado del mundo y no sobre el texto del "
        f"agente.\n"
    )

    add("\n## 5. Dimensión cualitativa: rúbrica del juez LLM\n")
    if veredictos:
        modelo_juez = next(
            (v.get("modelo_juez") for v in veredictos.values() if v and "error" not in v),
            "?",
        )
        add(
            f"Juez: `{modelo_juez}`, distinto del actor. Recibe la traza y la "
            f"respuesta final, pero no si la corrida cumplió la meta. Escala 1 "
            f"(muy malo) a 5 (muy bueno).\n\n"
        )
        filas = []
        for exp, rs in agrupar(runs, "experiment").items():
            vs = [veredictos.get(r["run_id"]) for r in rs]
            vs = [v for v in vs if v and "error" not in v]
            if not vs:
                continue
            prom = promedios_rubrica(vs)
            filas.append(
                [exp, str(len(vs))] + [_opt(prom[d], "{:.2f}") for d in DIMENSIONES]
            )
        add(
            _tabla(
                ["configuración", "n juzgadas"]
                + [d.replace("_", " ") for d in DIMENSIONES],
                filas,
            )
        )
    else:
        add("_(no se ejecutó el juez)_\n")

    add("\n## 6. Experimentos\n")
    add(
        "Cada fila cambia una sola cosa respecto del baseline. `MyAgent` es el "
        "mismo en todas: lo que cambia es la configuración que recibe de "
        "`build_agent`.\n\n"
    )
    orden = ["baseline"] + [k for k in EXPERIMENTS if k != "baseline"]
    presentes = [e for e in orden if any(r["experiment"] == e for r in runs)]
    add(
        _tabla(
            _CABECERA_AGREGADO,
            [
                _fila_agregado(e, agregar([r for r in runs if r["experiment"] == e], e))
                for e in presentes
            ],
        )
    )

    add("\n### Qué cambia cada configuración\n\n")
    for exp in presentes:
        cfg = EXPERIMENTS.get(exp)
        if cfg:
            add(f"- **`{exp}`** — {cfg.description}\n")

    add("\n### Efecto por dificultad\n\n")
    dificultades = sorted(
        {r["difficulty"] for r in runs}, key=lambda d: DIFFICULTY_ORDER.get(d, 99)
    )
    filas = []
    for exp in presentes:
        rs = [r for r in runs if r["experiment"] == exp]
        fila = [exp]
        for dif in dificultades:
            sub = [r for r in rs if r["difficulty"] == dif]
            fila.append(_pct(agregar(sub, dif).goal_rate) if sub else "—")
        filas.append(fila)
    add(_tabla(["configuración"] + dificultades, filas))

    return "\n".join(partes)


def escribir_informe(
    path: Path,
    runs: list[dict[str, Any]],
    veredictos: dict[str, dict[str, Any]] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(construir_informe(runs, veredictos), encoding="utf-8")
    return path
