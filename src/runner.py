"""Orquestrador da suíte de experimentos e geração das bases de dados CSV."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.approximations import christofides, twice_around_the_tree
from src.branch_and_bound import solve_branch_and_bound
from src.features import extract_instance_features
from src.heuristics import cold_start, convex_hull_insertion, nearest_neighbor
from src.tsplib_parser import TSPInstance, list_tsplib_instances
from src.types import TourResult


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
    mem_res = solve_branch_and_bound(
        dist_matrix,
        initial_tour=init_tour,
        initial_bound=init_bound,
        strategy_name=strategy,
        timeout_sec=min(timeout_sec, 2.0),
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
    }


def run_experiments(
    data_dir: Path = Path("data/tsplib"),
    output_dir: Path = Path("results"),
    timeout_sec: float = 1800.0,
    workers: int = 1,
    lb_modes: Optional[List[str]] = None,
    max_n: Optional[int] = None,
    skip_bnb: bool = False,
) -> None:
    """Executa features, heurísticas/aproximados e Branch-and-Bound sobre as instâncias TSPLIB."""
    output_dir.mkdir(parents=True, exist_ok=True)
    modes = lb_modes if lb_modes is not None else ["mst"]
    strategies = [
        "cold_start",
        "nearest_neighbor",
        "convex_hull_insertion",
        "twice_around_the_tree",
        "christofides",
    ]

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

    # 2. Heurísticas e algoritmos aproximados
    approx_path = output_dir / "approximations_results.csv"
    approx_fields = [
        "instance",
        "n",
        "algorithm",
        "cost",
        "elapsed_time_s",
        "peak_memory_bytes",
        "subroutine_times_json",
    ]
    warm_starts: Dict[Tuple[str, str], TourResult] = {}

    with approx_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=approx_fields)
        writer.writeheader()

        for inst in instances:
            for strat in strategies:
                res = _run_heuristic_or_approx(strat, inst.coords, inst.dist_matrix, track_memory=False)
                mem_res = _run_heuristic_or_approx(strat, inst.coords, inst.dist_matrix, track_memory=True)
                warm_starts[(inst.name, strat)] = res

                writer.writerow(
                    {
                        "instance": inst.name,
                        "n": inst.dimension,
                        "algorithm": strat,
                        "cost": f"{res.cost:.2f}",
                        "elapsed_time_s": f"{res.elapsed_time:.6f}",
                        "peak_memory_bytes": mem_res.peak_memory_bytes,
                        "subroutine_times_json": json.dumps(res.subroutine_times, sort_keys=True),
                    }
                )
            f.flush()

    if skip_bnb:
        return

    # 3. Branch-and-Bound com as 5 estratégias de warm-start
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
    ]

    tasks: List[Tuple[str, np.ndarray, List[int], float, str, str, float]] = []
    for inst in instances:
        for lb_mode in modes:
            for strat in strategies:
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

    with bnb_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=bnb_fields)
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
    args = parser.parse_args()

    run_experiments(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        timeout_sec=args.timeout,
        workers=args.workers,
        lb_modes=args.lb_modes,
        max_n=args.max_n,
        skip_bnb=args.skip_bnb,
    )


if __name__ == "__main__":
    main()
