"""Punto de entrada de la evaluación de M3.

    python eval/run.py                    barrido completo + juez + informe
    python eval/run.py --solo-baseline    solo la configuración de referencia
    python eval/run.py --estimar          coste estimado, sin llamar al modelo
    python eval/run.py --solo-informe     regenera el informe desde los jsonl
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mia_world import list_scenarios  # noqa: E402

from eval.configs import (  # noqa: E402
    DEFAULT_EXPERIMENTS,
    DIFFICULTY_ORDER,
    EXPERIMENTS,
    OPTIMAL_CALLS,
    PRICING,
    RESULTS_DIR,
    SCENARIOS_DIR,
    estimate_cost,
)
from eval.harness import preflight, run_once  # noqa: E402
from eval.metrics import escribir_jsonl, leer_jsonl  # noqa: E402
from eval.report import escribir_informe  # noqa: E402

RUNS_PATH = RESULTS_DIR / "runs.jsonl"
JUDGMENTS_PATH = RESULTS_DIR / "judgments.jsonl"
REPORT_PATH = RESULTS_DIR / "report.md"


def _cargar_escenarios(filtro: list[str] | None) -> list:
    escenarios = list_scenarios(SCENARIOS_DIR)
    if filtro:
        pedidos = set(filtro)
        escenarios = [
            s for s in escenarios if s.id in pedidos or s.difficulty in pedidos
        ]
        if not escenarios:
            raise SystemExit(f"Ningún escenario coincide con {filtro}.")
    return sorted(
        escenarios,
        key=lambda s: (
            DIFFICULTY_ORDER.get(s.difficulty, 99),
            OPTIMAL_CALLS.get(s.id, 0),
        ),
    )


def _estimar(escenarios: list, experimentos: list[str], reps: int) -> None:
    """Techo de coste: asume que toda corrida agota su presupuesto de pasos.

    El prompt se reenvía completo en cada iteración, así que la entrada
    crece con el cuadrado de los pasos.
    """
    modelo = os.environ.get("BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0")
    base_tokens = 900
    por_paso = 120
    total_in = total_out = n_runs = 0

    for exp_name in experimentos:
        exp = EXPERIMENTS[exp_name]
        for sc in escenarios:
            pasos = exp.step_budget(sc.id)
            entrada = sum(base_tokens + por_paso * i for i in range(1, pasos + 1))
            total_in += entrada * reps
            total_out += pasos * 60 * reps
            n_runs += reps

    tarifado = "sí" if modelo in PRICING else "NO (modelo sin tarifa conocida)"
    print(f"Modelo:            {modelo}  (tarifado: {tarifado})")
    print(
        f"Corridas:          {n_runs}  ({len(escenarios)} escenarios x "
        f"{len(experimentos)} configs x {reps} reps)"
    )
    print(f"Tokens estimados:  {total_in:,} in / {total_out:,} out")
    print(f"Coste estimado:    ${estimate_cost(modelo, total_in, total_out):.3f}")
    print()
    print("Es un techo. Las corridas que resuelven temprano cuestan menos.")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="eval/run.py", description=__doc__)
    p.add_argument("--reps", type=int, default=3,
                   help="Repeticiones por celda (defecto: 3).")
    p.add_argument("--experimentos", nargs="*", default=None,
                   help=f"Subconjunto de: {', '.join(EXPERIMENTS)}.")
    p.add_argument("--escenarios", nargs="*", default=None,
                   help="Ids o dificultades. Por defecto, los ocho.")
    p.add_argument("--solo-baseline", action="store_true",
                   help="Atajo para --experimentos baseline.")
    p.add_argument("--sin-juez", action="store_true",
                   help="Salta la rúbrica LLM.")
    p.add_argument("--solo-informe", action="store_true",
                   help="Regenera report.md desde los jsonl.")
    p.add_argument("--estimar", action="store_true",
                   help="Muestra el coste estimado y sale.")
    p.add_argument("--preflight", action="store_true",
                   help="Valida las credenciales del proveedor y sale.")
    args = p.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.preflight:
        ok, detalle = preflight()
        print(("OK  " if ok else "FALLO  ") + detalle)
        return 0 if ok else 1

    if args.solo_informe:
        runs = leer_jsonl(RUNS_PATH)
        veredictos = {v["run_id"]: v["veredicto"] for v in leer_jsonl(JUDGMENTS_PATH)}
        escribir_informe(REPORT_PATH, runs, veredictos)
        print(f"Informe regenerado: {REPORT_PATH}  ({len(runs)} corridas)")
        return 0

    experimentos = (
        ["baseline"]
        if args.solo_baseline
        else (args.experimentos or DEFAULT_EXPERIMENTS)
    )
    desconocidos = [e for e in experimentos if e not in EXPERIMENTS]
    if desconocidos:
        raise SystemExit(f"Experimentos desconocidos: {desconocidos}")

    escenarios = _cargar_escenarios(args.escenarios)

    if args.estimar:
        _estimar(escenarios, experimentos, args.reps)
        return 0

    ok, detalle = preflight()
    if not ok:
        print(f"Preflight falló, no se lanza el barrido:\n  {detalle}", file=sys.stderr)
        return 1
    print(f"Preflight OK ({detalle}).\n")

    total = len(escenarios) * len(experimentos) * args.reps
    print(
        f"Barrido: {total} corridas ({len(escenarios)} escenarios x "
        f"{len(experimentos)} configs x {args.reps} reps)\n"
    )

    registros = []
    hecho = 0
    for exp_name in experimentos:
        exp = EXPERIMENTS[exp_name]
        for sc in escenarios:
            for rep in range(args.reps):
                hecho += 1
                rec = run_once(sc, exp, rep)
                registros.append(rec.to_json())
                marca = "OK " if rec.goal_achieved else "-- "
                print(
                    f"[{hecho:>3}/{total}] {marca} {exp_name:<14} {sc.id:<22} "
                    f"rep{rep}  calls={rec.n_calls:>2}/{rec.step_budget:<2} "
                    f"err={rec.n_error_calls:<2} ${rec.cost_usd:.4f} "
                    f"{rec.wall_seconds:>5.1f}s"
                )
                # Se persiste en cada vuelta para no perder lo ya pagado si el
                # barrido se corta.
                escribir_jsonl(RUNS_PATH, registros)

    coste = sum(r["cost_usd"] for r in registros)
    exitos = sum(1 for r in registros if r["goal_achieved"])
    print(
        f"\nBarrido terminado: {exitos}/{len(registros)} corridas resueltas. "
        f"Coste ${coste:.4f}. Trazas en {RUNS_PATH}"
    )

    veredictos: dict[str, dict] = {}
    if not args.sin_juez:
        from eval.judge import construir_juez, juzgar

        print(f"\nJuez LLM sobre {len(registros)} corridas...")
        juez = construir_juez()
        mensajes = {s.id: s.user_message for s in escenarios}
        filas = []
        for i, rec in enumerate(registros, 1):
            v = juzgar(rec, mensajes.get(rec["scenario_id"], ""), juez)
            veredictos[rec["run_id"]] = v
            filas.append({"run_id": rec["run_id"], "veredicto": v})
            estado = "error" if "error" in v else "ok"
            print(f"  [{i:>3}/{len(registros)}] {rec['scenario_id']:<22} {estado}")
            escribir_jsonl(JUDGMENTS_PATH, filas)

    escribir_informe(REPORT_PATH, registros, veredictos)
    print(f"\nInforme: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
