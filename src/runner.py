"""Orquestrador da suíte de experimentos e geração das bases de dados CSV."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import statistics
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from src.approximations import christofides, twice_around_the_tree
from src.branch_and_bound import solve_branch_and_bound
from src.features import extract_instance_features
from src.heuristics import cold_start, convex_hull_insertion, nearest_neighbor
from src.tsplib_parser import TSPInstance, list_tsplib_instances
from src.types import TourResult

_STRATEGIES: List[str] = [
    "cold_start",
    "nearest_neighbor",
    "convex_hull_insertion",
    "twice_around_the_tree",
    "christofides",
]


def _run_heuristic_or_approx(
    name: str,
    coords: np.ndarray,
    dist_matrix: np.ndarray,
    track_memory: bool = False,
) -> TourResult:
    if name == "cold_start":
        return cold_start(dist_matrix, track_memory=track_memory)
    if name == "nearest_neighbor":
        return nearest_neighbor(dist_matrix, start=0, track_memory=track_memory)
    if name == "convex_hull_insertion":
        return convex_hull_insertion(coords, dist_matrix, track_memory=track_memory)
    if name == "twice_around_the_tree":
        return twice_around_the_tree(dist_matrix, root=0, track_memory=track_memory)
    if name == "christofides":
        return christofides(dist_matrix, track_memory=track_memory)
    raise ValueError(f"Estratégia desconhecida: {name!r}")


def _benchmark_heuristic_median(
    name: str,
    coords: np.ndarray,
    dist_matrix: np.ndarray,
    repeats: int = 15,
) -> Tuple[TourResult, int, Dict[str, int]]:
    """Executa warm-up + múltiplas repetições (mediana) para estabilizar tempos e mede memória isolada."""
    # Warm-up
    _run_heuristic_or_approx(name, coords, dist_matrix, track_memory=False)

    n_runs = max(1, int(repeats))
    runs: List[TourResult] = [
        _run_heuristic_or_approx(name, coords, dist_matrix, track_memory=False)
        for _ in range(n_runs)
    ]

    median_elapsed = float(statistics.median(r.elapsed_time for r in runs))
    sub_keys = list(runs[0].subroutine_times.keys())
    median_sub_times: Dict[str, float] = {
        k: float(statistics.median(r.subroutine_times.get(k, 0.0) for r in runs))
        for k in sub_keys
    }

    mem_res = _run_heuristic_or_approx(name, coords, dist_matrix, track_memory=True)
    stabilized = TourResult(
        tour=runs[0].tour,
        cost=runs[0].cost,
        elapsed_time=median_elapsed,
        subroutine_times=median_sub_times,
        subroutine_memory_bytes=mem_res.subroutine_memory_bytes,
        peak_memory_bytes=mem_res.peak_memory_bytes,
    )
    return stabilized, mem_res.peak_memory_bytes, mem_res.subroutine_memory_bytes


def _run_bnb_task(task: Tuple[str, np.ndarray, List[int], float, str, str, float]) -> Dict[str, Any]:
    """Executa uma tarefa individual de Branch-and-Bound (compatível com multiprocessing)."""
    inst_name, dist_matrix, init_tour, init_bound, strategy, lb_mode, timeout_sec = task
    n = dist_matrix.shape[0]

    res = solve_branch_and_bound(
        dist_matrix,
        initial_tour=init_tour,
        initial_bound=init_bound,
        strategy_name=strategy,
        timeout_sec=timeout_sec,
        track_memory=False,
        lower_bound_mode=lb_mode,
    )
    # Como o DFS tem memória O(N) com buffers pré-alocados, uma sonda curta captura o pico real
    mem_res = solve_branch_and_bound(
        dist_matrix,
        initial_tour=init_tour,
        initial_bound=init_bound,
        strategy_name=strategy,
        timeout_sec=min(timeout_sec, 0.05),
        track_memory=True,
        lower_bound_mode=lb_mode,
    )

    return {
        "instance": inst_name,
        "n": n,
        "lb_mode": lb_mode,
        "strategy": strategy,
        "timed_out": res.timed_out,
        "elapsed_time_s": f"{res.elapsed_time:.6f}" if res.elapsed_time is not None else "NA",
        "nodes_expanded": res.nodes_expanded,
        "nodes_pruned": res.nodes_pruned,
        "max_depth": res.max_depth,
        "initial_bound": f"{res.initial_bound:.2f}" if np.isfinite(res.initial_bound) else "inf",
        "best_cost": f"{res.best_cost:.2f}",
        "precompute_s": f"{res.subroutine_times.get('precompute_bounds', 0.0):.6f}",
        "search_loop_s": f"{res.subroutine_times.get('search_loop', 0.0):.6f}",
        "peak_memory_bytes": mem_res.peak_memory_bytes,
        "subroutine_memory_json": json.dumps(mem_res.subroutine_memory_bytes, sort_keys=True),
    }


def _sort_bnb_csv(bnb_path: Path, fieldnames: List[str]) -> None:
    """Ordena bnb_results.csv deterministicamente por (n, instance, lb_mode, strategy)."""
    if not bnb_path.exists():
        return
    with bnb_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    strat_order = {s: idx for idx, s in enumerate(_STRATEGIES)}
    rows.sort(
        key=lambda r: (
            int(r["n"]),
            str(r["instance"]),
            str(r["lb_mode"]),
            strat_order.get(str(r["strategy"]), 99),
        )
    )
    with bnb_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_experiments(
    data_dir: Path = Path("data/tsplib"),
    output_dir: Path = Path("results"),
    timeout_sec: float = 1800.0,
    workers: int = 1,
    lb_modes: Optional[List[str]] = None,
    max_n: Optional[int] = None,
    skip_bnb: bool = False,
    repeats: int = 15,
    resume: bool = True,
) -> None:
    """Executa features, heurísticas/aproximados e Branch-and-Bound sobre as instâncias TSPLIB."""
    output_dir.mkdir(parents=True, exist_ok=True)
    modes = lb_modes if lb_modes is not None else ["mst"]

    instances: List[TSPInstance] = list_tsplib_instances(data_dir)
    if max_n is not None:
        instances = [inst for inst in instances if inst.dimension <= max_n]

    # 1. Extração de features
    features_path = output_dir / "instances_features.csv"
    feature_rows: List[Dict[str, Any]] = []
    for inst in instances:
        feats = extract_instance_features(inst.coords, inst.dist_matrix)
        row: Dict[str, Any] = {
            "instance": inst.name,
            "n": inst.dimension,
            "optimal_cost": inst.optimal_cost if inst.optimal_cost is not None else "NA",
        }
        row.update(feats)
        feature_rows.append(row)

    if feature_rows:
        with features_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(feature_rows[0].keys()))
            writer.writeheader()
            writer.writerows(feature_rows)

    # 2. Heurísticas e algoritmos aproximados (com warm-up e mediana de repetições)
    approx_path = output_dir / "approximations_results.csv"
    approx_fields = [
        "instance",
        "n",
        "algorithm",
        "cost",
        "elapsed_time_s",
        "peak_memory_bytes",
        "subroutine_times_json",
        "subroutine_memory_json",
    ]
    warm_starts: Dict[Tuple[str, str], TourResult] = {}

    with approx_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=approx_fields)
        writer.writeheader()

        for inst in instances:
            for strat in _STRATEGIES:
                res, peak_mem, sub_mem = _benchmark_heuristic_median(
                    strat, inst.coords, inst.dist_matrix, repeats=repeats
                )
                warm_starts[(inst.name, strat)] = res

                writer.writerow(
                    {
                        "instance": inst.name,
                        "n": inst.dimension,
                        "algorithm": strat,
                        "cost": f"{res.cost:.2f}",
                        "elapsed_time_s": f"{res.elapsed_time:.6f}",
                        "peak_memory_bytes": peak_mem,
                        "subroutine_times_json": json.dumps(res.subroutine_times, sort_keys=True),
                        "subroutine_memory_json": json.dumps(sub_mem, sort_keys=True),
                    }
                )
            f.flush()

    if skip_bnb:
        return

    # 3. Branch-and-Bound com checkpoint/resume e ordenação determinística
    bnb_path = output_dir / "bnb_results.csv"
    bnb_fields = [
        "instance",
        "n",
        "lb_mode",
        "strategy",
        "timed_out",
        "elapsed_time_s",
        "nodes_expanded",
        "nodes_pruned",
        "max_depth",
        "initial_bound",
        "best_cost",
        "precompute_s",
        "search_loop_s",
        "peak_memory_bytes",
        "subroutine_memory_json",
    ]

    completed_keys: Set[Tuple[str, str, str]] = set()
    if resume and bnb_path.exists():
        with bnb_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames == bnb_fields:
                for row in reader:
                    completed_keys.add((row["instance"], row["lb_mode"], row["strategy"]))

    tasks: List[Tuple[str, np.ndarray, List[int], float, str, str, float]] = []
    for inst in instances:
        for lb_mode in modes:
            for strat in _STRATEGIES:
                if (inst.name, lb_mode, strat) in completed_keys:
                    continue
                ws = warm_starts[(inst.name, strat)]
                tasks.append(
                    (
                        inst.name,
                        inst.dist_matrix,
                        ws.tour,
                        ws.cost,
                        strat,
                        lb_mode,
                        timeout_sec,
                    )
                )

    open_mode = "a" if (resume and completed_keys) else "w"
    with bnb_path.open(open_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=bnb_fields)
        if open_mode == "w":
            writer.writeheader()

        if workers <= 1:
            for task in tasks:
                row = _run_bnb_task(task)
                writer.writerow(row)
                f.flush()
                print(
                    f"[BnB] {row['instance']} (N={row['n']}) | {row['lb_mode']} | "
                    f"{row['strategy']} -> time={row['elapsed_time_s']} | "
                    f"expanded={row['nodes_expanded']} | best={row['best_cost']}"
                )
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_run_bnb_task, task) for task in tasks]
                for fut in as_completed(futures):
                    row = fut.result()
                    writer.writerow(row)
                    f.flush()
                    print(
                        f"[BnB] {row['instance']} (N={row['n']}) | {row['lb_mode']} | "
                        f"{row['strategy']} -> time={row['elapsed_time_s']} | "
                        f"expanded={row['nodes_expanded']} | best={row['best_cost']}"
                    )

    _sort_bnb_csv(bnb_path, bnb_fields)


def main() -> None:
    parser = argparse.ArgumentParser(description="Runner experimental do TP2 (TSP).")
    parser.add_argument("--data-dir", type=Path, default=Path("data/tsplib"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--timeout", type=float, default=1800.0, help="Timeout do B&B em segundos.")
    parser.add_argument("--workers", type=int, default=1, help="Processos paralelos para o B&B.")
    parser.add_argument(
        "--lb-modes",
        nargs="+",
        default=["mst"],
        choices=["two_min", "mst"],
        help="Modos de lower bound para o B&B.",
    )
    parser.add_argument("--max-n", type=int, default=None, help="Filtra instâncias com N <= max_n.")
    parser.add_argument("--skip-bnb", action="store_true", help="Executa apenas features e aproximados.")
    parser.add_argument(
        "--repeats",
        type=int,
        default=15,
        help="Número de repetições (mediana) para heurísticas e aproximados.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Desativa retomada por checkpoint e sobrescreve bnb_results.csv.",
    )
    args = parser.parse_args()

    run_experiments(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        timeout_sec=args.timeout,
        workers=args.workers,
        lb_modes=args.lb_modes,
        max_n=args.max_n,
        skip_bnb=args.skip_bnb,
        repeats=args.repeats,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
