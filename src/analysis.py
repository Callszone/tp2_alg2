"""Modelagem estatística de complexidade empírica e análise de sensibilidade."""

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


_COMPLEXITY_BASES: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "O(N)": lambda n: n,
    "O(N log N)": lambda n: n * np.log2(np.maximum(n, 2.0)),
    "O(N^2)": lambda n: n ** 2,
    "O(N^3)": lambda n: n ** 3,
}

_FEATURE_COLS: List[str] = [
    "aspect_ratio",
    "convex_hull_ratio",
    "spatial_entropy",
    "mst_max_degree",
    "mst_leaf_ratio",
    "nn_dist_cv",
]


def _fit_complexity_curves(n_vals: np.ndarray, y_vals: np.ndarray) -> List[Dict[str, float | str]]:
    """Ajusta y ~ a * f(N) + b para cada curva teórica e retorna métricas R2 e RMSE."""
    results: List[Dict[str, float | str]] = []
    if len(n_vals) < 2:
        return results

    for label, basis_fn in _COMPLEXITY_BASES.items():
        x_feat = basis_fn(n_vals).reshape(-1, 1)
        reg = LinearRegression().fit(x_feat, y_vals)
        preds = reg.predict(x_feat)
        r2 = float(r2_score(y_vals, preds))
        rmse = float(np.sqrt(mean_squared_error(y_vals, preds)))
        results.append(
            {
                "model": label,
                "coef": float(reg.coef_[0]),
                "intercept": float(reg.intercept_),
                "r2": r2,
                "rmse": rmse,
            }
        )
    return results


def _build_complexity_and_subroutines(
    approx_df: pd.DataFrame,
    results_dir: Path,
    figures_dir: Path,
) -> pd.DataFrame:
    """Ajusta curvas assintóticas (tempo e memória) e extrai breakdown de subrotinas."""
    reg_rows: List[Dict[str, object]] = []
    for alg, grp in approx_df.groupby("algorithm"):
        n_vals = grp["n"].to_numpy(dtype=np.float64)
        for metric_col, target_name in [
            ("elapsed_time_s", "time_s"),
            ("peak_memory_bytes", "memory_bytes"),
        ]:
            y_vals = grp[metric_col].to_numpy(dtype=np.float64)
            for fit in _fit_complexity_curves(n_vals, y_vals):
                reg_rows.append({"algorithm": alg, "target": target_name, **fit})

    reg_df = pd.DataFrame(reg_rows)
    reg_df.to_csv(results_dir / "complexity_regression.csv", index=False)

    # Breakdown de subrotinas
    sub_rows: List[Dict[str, object]] = []
    for _, row in approx_df.iterrows():
        timings = json.loads(row["subroutine_times_json"])
        total_sub = sum(timings.values()) if timings else 0.0
        for sub_name, sub_t in timings.items():
            sub_rows.append(
                {
                    "instance": row["instance"],
                    "n": int(row["n"]),
                    "algorithm": row["algorithm"],
                    "subroutine": sub_name,
                    "time_s": float(sub_t),
                    "share_pct": (100.0 * float(sub_t) / total_sub) if total_sub > 0 else 0.0,
                }
            )
    sub_df = pd.DataFrame(sub_rows)
    sub_df.to_csv(results_dir / "subroutine_breakdown.csv", index=False)

    # Gráfico 1: Tempo vs N dos algoritmos aproximados e heurísticas
    fig, ax = plt.subplots(figsize=(8, 5))
    for alg, grp in approx_df.groupby("algorithm"):
        g_sorted = grp.sort_values("n")
        ax.plot(g_sorted["n"], g_sorted["elapsed_time_s"], marker="o", label=alg)
    ax.set_xlabel("Número de vértices (N)")
    ax.set_ylabel("Tempo de execução (s)")
    ax.set_title("Escalabilidade Empírica: Heurísticas e Aproximados")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "complexity_time.png", dpi=150)
    plt.close(fig)

    # Gráfico 2: Subrotinas do Christofides (isolando Matching vs MST)
    chris_sub = sub_df[sub_df["algorithm"] == "christofides"]
    if not chris_sub.empty:
        piv = chris_sub.pivot_table(
            index="n", columns="subroutine", values="share_pct", aggfunc="mean"
        ).fillna(0.0)
        fig, ax = plt.subplots(figsize=(9, 5))
        piv.plot(kind="bar", stacked=True, ax=ax)
        ax.set_xlabel("Número de vértices (N)")
        ax.set_ylabel("Participação no tempo (%)")
        ax.set_title("Profiling de Subrotinas no Algoritmo de Christofides")
        ax.legend(title="Subrotina", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(figures_dir / "christofides_subroutines.png", dpi=150)
        plt.close(fig)

    return reg_df


def _build_warmstart_summary(
    bnb_df: pd.DataFrame,
    approx_df: pd.DataFrame,
    results_dir: Path,
    figures_dir: Path,
) -> pd.DataFrame:
    """Consolida impacto do warm-start na poda e no tempo total."""
    if bnb_df.empty:
        return pd.DataFrame()

    merged = bnb_df.merge(
        approx_df[["instance", "algorithm", "elapsed_time_s"]].rename(
            columns={"algorithm": "strategy", "elapsed_time_s": "heuristic_time_s"}
        ),
        on=["instance", "strategy"],
        how="left",
    )
    merged["bnb_time_s"] = pd.to_numeric(merged["elapsed_time_s"], errors="coerce")
    merged["total_time_s"] = merged["bnb_time_s"] + merged["heuristic_time_s"]
    total_visited = merged["nodes_expanded"] + merged["nodes_pruned"]
    merged["prune_ratio"] = np.where(
        total_visited > 0, merged["nodes_pruned"] / total_visited, 0.0
    )

    cold_nodes = (
        merged[merged["strategy"] == "cold_start"]
        .set_index(["instance", "lb_mode"])["nodes_expanded"]
        .to_dict()
    )
    merged["nodes_vs_cold_ratio"] = [
        row["nodes_expanded"] / cold_nodes[(row["instance"], row["lb_mode"])]
        if (row["instance"], row["lb_mode"]) in cold_nodes
        and cold_nodes[(row["instance"], row["lb_mode"])] > 0
        else np.nan
        for _, row in merged.iterrows()
    ]

    merged.to_csv(results_dir / "warmstart_summary.csv", index=False)

    # Gráfico 3: Nós expandidos por estratégia nas instâncias sem timeout
    solved = merged[~merged["timed_out"].astype(bool)]
    if not solved.empty:
        piv = solved.pivot_table(
            index="n", columns="strategy", values="nodes_expanded", aggfunc="mean"
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        piv.plot(kind="bar", ax=ax)
        ax.set_xlabel("Número de vértices (N)")
        ax.set_ylabel("Nós expandidos (escala log)")
        ax.set_yscale("log")
        ax.set_title("Impacto do Warm-Start na Árvore do Branch-and-Bound")
        ax.legend(title="Estratégia", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(figures_dir / "warmstart_nodes.png", dpi=150)
        plt.close(fig)

    return merged


def _build_sensitivity_analysis(
    features_df: pd.DataFrame,
    approx_df: pd.DataFrame,
    bnb_df: pd.DataFrame,
    results_dir: Path,
    figures_dir: Path,
) -> pd.DataFrame:
    """Calcula gap de aproximação e treina modelos de sensibilidade sobre as features."""
    ref_costs: Dict[str, Tuple[float, str]] = {}
    for _, row in features_df.iterrows():
        inst = str(row["instance"])
        opt = pd.to_numeric(row["optimal_cost"], errors="coerce")
        if pd.notna(opt):
            ref_costs[inst] = (float(opt), "tsplib_exact")
            continue

        if not bnb_df.empty:
            sub = bnb_df[bnb_df["instance"] == inst]
            exact_sub = sub[~sub["timed_out"].astype(bool)]
            if not exact_sub.empty:
                ref_costs[inst] = (float(exact_sub["best_cost"].min()), "bnb_exact")
                continue
            if not sub.empty:
                ref_costs[inst] = (float(sub["best_cost"].min()), "bnb_best_partial")
                continue

        min_approx = float(approx_df[approx_df["instance"] == inst]["cost"].min())
        ref_costs[inst] = (min_approx, "best_approx")

    gap_df = approx_df.merge(features_df, on=["instance", "n"], how="left")
    gap_df["ref_cost"] = [ref_costs[str(inst)][0] for inst in gap_df["instance"]]
    gap_df["ref_source"] = [ref_costs[str(inst)][1] for inst in gap_df["instance"]]
    gap_df["gap_pct"] = 100.0 * (gap_df["cost"] - gap_df["ref_cost"]) / gap_df["ref_cost"]
    gap_df.to_csv(results_dir / "gap_by_instance.csv", index=False)

    # Análise de sensibilidade focada nos aproximados e heurísticas não-triviais
    target_algs = [
        "nearest_neighbor",
        "convex_hull_insertion",
        "twice_around_the_tree",
        "christofides",
    ]
    imp_rows: List[Dict[str, object]] = []

    for alg in target_algs:
        sub = gap_df[gap_df["algorithm"] == alg].dropna(subset=_FEATURE_COLS + ["gap_pct"])
        if len(sub) < 3:
            continue

        x_mat = sub[_FEATURE_COLS].replace([np.inf, -np.inf], np.nan).dropna()
        if len(x_mat) < 3:
            continue
        y_vec = sub.loc[x_mat.index, "gap_pct"].to_numpy(dtype=np.float64)

        x_scaled = StandardScaler().fit_transform(x_mat.to_numpy(dtype=np.float64))
        lin = LinearRegression().fit(x_scaled, y_vec)
        lin_r2 = float(r2_score(y_vec, lin.predict(x_scaled)))

        tree = DecisionTreeRegressor(max_depth=3, random_state=42).fit(
            x_mat.to_numpy(dtype=np.float64), y_vec
        )
        tree_r2 = float(r2_score(y_vec, tree.predict(x_mat.to_numpy(dtype=np.float64))))

        for idx, feat_name in enumerate(_FEATURE_COLS):
            imp_rows.append(
                {
                    "algorithm": alg,
                    "feature": feat_name,
                    "tree_importance": float(tree.feature_importances_[idx]),
                    "linear_std_coef": float(lin.coef_[idx]),
                    "linear_r2": lin_r2,
                    "tree_r2": tree_r2,
                }
            )

    imp_df = pd.DataFrame(imp_rows)
    imp_df.to_csv(results_dir / "sensitivity_feature_importance.csv", index=False)

    if not imp_df.empty:
        piv = imp_df.pivot(index="feature", columns="algorithm", values="tree_importance").fillna(0.0)
        fig, ax = plt.subplots(figsize=(9, 5))
        piv.plot(kind="barh", ax=ax)
        ax.set_xlabel("Importância na Árvore de Decisão")
        ax.set_ylabel("Feature Geométrica/Topológica")
        ax.set_title("Análise de Sensibilidade: Impacto das Features no Gap (%)")
        ax.legend(title="Algoritmo", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(figures_dir / "sensitivity_importance.png", dpi=150)
        plt.close(fig)

    return imp_df


def run_analysis(
    results_dir: Path = Path("results"),
    figures_dir: Path = Path("report/figures"),
) -> None:
    """Executa modelagem de complexidade, warm-start e sensibilidade sobre os CSVs."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    features_path = results_dir / "instances_features.csv"
    approx_path = results_dir / "approximations_results.csv"
    bnb_path = results_dir / "bnb_results.csv"

    if not features_path.exists() or not approx_path.exists():
        raise FileNotFoundError(
            f"CSVs de entrada não encontrados em {results_dir}. Execute src.runner primeiro."
        )

    features_df = pd.read_csv(features_path)
    approx_df = pd.read_csv(approx_path)
    bnb_df = pd.read_csv(bnb_path) if bnb_path.exists() else pd.DataFrame()

    _build_complexity_and_subroutines(approx_df, results_dir, figures_dir)
    _build_warmstart_summary(bnb_df, approx_df, results_dir, figures_dir)
    _build_sensitivity_analysis(features_df, approx_df, bnb_df, results_dir, figures_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Análise estatística e gráficos do TP2.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("report/figures"))
    args = parser.parse_args()
    run_analysis(results_dir=args.results_dir, figures_dir=args.figures_dir)


if __name__ == "__main__":
    main()
