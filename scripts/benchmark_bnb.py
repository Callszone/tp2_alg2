"""Benchmark reprodutível do Branch-and-Bound em instâncias sintéticas EUC_2D."""

import argparse
import csv
from pathlib import Path
from typing import List
import numpy as np

from src.approximations import twice_around_the_tree
from src.branch_and_bound import solve_branch_and_bound
from src.tsplib_parser import compute_euc2d_matrix


def make_euc2d_instance(n: int, seed: int = 42) -> np.ndarray:
    """Gera matriz de distâncias EUC_2D (arredondamento nint TSPLIB) em [0, 1000]^2."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, 1000.0, size=(n, 2))
    return compute_euc2d_matrix(coords)


def run_benchmark(
    sizes: List[int],
    lb_modes: List[str],
    timeout_sec: float,
    seed: int,
    output_path: Path,
) -> None:
    """Executa o benchmark comparando modos de lower bound e grava os resultados em CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "n",
        "seed",
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

    print(
        f"{'N':>3} | {'LB':<7} | {'Strategy':<10} | {'Time(s)':>8} | {'Expanded':>10} | "
        f"{'Pruned':>10} | {'BestCost':>9} | {'PeakMem(KB)':>11}"
    )
    print("-" * 88)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for n in sizes:
            dist = make_euc2d_instance(n, seed=seed)
            tatt = twice_around_the_tree(dist)

            configs = [
                ("cold_start", None, None),
                ("tatt_warm", tatt.tour, tatt.cost),
            ]
            for lb_mode in lb_modes:
                for strat, tour, bound in configs:
                    res = solve_branch_and_bound(
                        dist,
                        initial_tour=tour,
                        initial_bound=bound,
                        strategy_name=strat,
                        timeout_sec=timeout_sec,
                        track_memory=False,
                        lower_bound_mode=lb_mode,
                    )
                    # Medição de memória isolada para não distorcer elapsed_time_s
                    mem_res = solve_branch_and_bound(
                        dist,
                        initial_tour=tour,
                        initial_bound=bound,
                        strategy_name=strat,
                        timeout_sec=min(timeout_sec, 1.0),
                        track_memory=True,
                        lower_bound_mode=lb_mode,
                    )

                    time_str = f"{res.elapsed_time:.6f}" if res.elapsed_time is not None else "NA"
                    init_b_str = f"{res.initial_bound:.2f}" if np.isfinite(res.initial_bound) else "inf"

                    writer.writerow(
                        {
                            "n": n,
                            "seed": seed,
                            "lb_mode": lb_mode,
                            "strategy": strat,
                            "timed_out": res.timed_out,
                            "elapsed_time_s": time_str,
                            "nodes_expanded": res.nodes_expanded,
                            "nodes_pruned": res.nodes_pruned,
                            "max_depth": res.max_depth,
                            "initial_bound": init_b_str,
                            "best_cost": f"{res.best_cost:.2f}",
                            "precompute_s": f"{res.subroutine_times.get('precompute_bounds', 0.0):.6f}",
                            "search_loop_s": f"{res.subroutine_times.get('search_loop', 0.0):.6f}",
                            "peak_memory_bytes": mem_res.peak_memory_bytes,
                        }
                    )

                    t_disp = f"{res.elapsed_time:8.4f}" if res.elapsed_time is not None else " TIMEOUT"
                    print(
                        f"{n:3d} | {lb_mode:<7} | {strat:<10} | {t_disp} | {res.nodes_expanded:10d} | "
                        f"{res.nodes_pruned:10d} | {res.best_cost:9.1f} | "
                        f"{mem_res.peak_memory_bytes / 1024:11.1f}"
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark do Branch-and-Bound para TSP.")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[10, 12, 14, 16, 18, 20, 24, 28, 32],
        help="Tamanhos N das instâncias sintéticas.",
    )
    parser.add_argument(
        "--lb-modes",
        nargs="+",
        default=["two_min", "mst"],
        choices=["two_min", "mst"],
        help="Modos de lower bound a comparar.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Timeout em segundos por execução (padrão: 5.0).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatória.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/benchmark_bnb.csv"),
        help="Caminho do CSV de saída.",
    )
    args = parser.parse_args()
    run_benchmark(args.sizes, args.lb_modes, args.timeout, args.seed, args.output)


if __name__ == "__main__":
    main()
