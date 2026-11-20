"""Modelagem estatística de complexidade empírica, warm-start e análise de sensibilidade."""

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


_COMPLEXITY_BASES: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "O(N)": lambda n: n,
    "O(N log N)": lambda n: n * np.log2(np.maximum(n, 2.0)),
    "O(N^2)": lambda n: n**2,
    "O(N^3)": lambda n: n**3,
}

_FEATURE_COLS: List[str] = [
    "aspect_ratio",
    "convex_hull_ratio",
    "spatial_entropy",
    "mst_max_degree",
    "mst_leaf_ratio",
    "nn_dist_cv",
]


def _parse_bool_series(series: pd.Series) -> pd.Series:
    """Converte série booleana/string ('True'/'False') de forma segura."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def _fit_complexity_curves(n_vals: np.ndarray, y_vals: np.ndarray) -> List[Dict[str, float | str]]:
    """Ajusta y ~ a * f(N) + b com restrição a >= 0 para cada curva teórica."""
    results: List[Dict[str, float | str]] = []
    if len(n_vals) < 2:
        return results

    for label, basis_fn in _COMPLEXITY_BASES.items():
        x_feat = basis_fn(n_vals).reshape(-1, 1)
        reg = LinearRegression(positive=True).fit(x_feat, y_vals)
        preds = reg.predict(x_feat)
        r2 = float(r2_score(y_vals, preds)) if np.std(y_vals) > 1e-15 else 1.0
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


def _fit_exponential_curve(n_vals: np.ndarray, y_vals: np.ndarray) -> List[Dict[str, float | str]]:
    """Ajusta log(y) ~ a * N + b (comportamento exponencial O(b^N)) para o Branch-and-Bound."""
    mask = y_vals > 0
    if np.sum(mask) < 2:
        return []
    n_pos = n_vals[mask]
    log_y = np.log(y_vals[mask])
    reg = LinearRegression(positive=True).fit(n_pos.reshape(-1, 1), log_y)
    pred_log = reg.predict(n_pos.reshape(-1, 1))
    r2_log = float(r2_score(log_y, pred_log)) if np.std(log_y) > 1e-15 else 1.0
    rmse_log = float(np.sqrt(mean_squared_error(log_y, pred_log)))
    return [
        {
            "model": "O(b^N) [log-linear]",
            "coef": float(reg.coef_[0]),
            "intercept": float(reg.intercept_),
            "r2": r2_log,
            "rmse": rmse_log,
        }
    ]


def _build_complexity_and_subroutines(
    approx_df: pd.DataFrame,
    bnb_df: pd.DataFrame,
    results_dir: Path,
    figures_dir: Path,
) -> pd.DataFrame:
    """Ajusta curvas assintóticas (algoritmos, subrotinas e B&B) e gera gráficos com curvas ajustadas."""
    reg_rows: List[Dict[str, object]] = []

    # 1. Complexidade de heurísticas e algoritmos aproximados
    for alg, grp in approx_df.groupby("algorithm"):
        n_vals = grp["n"].to_numpy(dtype=np.float64)
        for metric_col, target_name in [
            ("elapsed_time_s", "time_s"),
            ("peak_memory_bytes", "memory_bytes"),
        ]:
            y_vals = grp[metric_col].to_numpy(dtype=np.float64)
            for fit in _fit_complexity_curves(n_vals, y_vals):
                reg_rows.append({"algorithm": str(alg), "target": target_name, **fit})

    # 2. Breakdown de subrotinas (tempo e memória)
    sub_rows: List[Dict[str, object]] = []
    has_sub_mem = "subroutine_memory_json" in approx_df.columns
    for _, row in approx_df.iterrows():
        timings: Dict[str, float] = json.loads(row["subroutine_times_json"])
        mem_dict: Dict[str, int] = (
            json.loads(row["subroutine_memory_json"])
            if has_sub_mem and pd.notna(row["subroutine_memory_json"])
            else {}
        )
        total_sub_t = sum(timings.values()) if timings else 0.0
        total_sub_m = sum(mem_dict.values()) if mem_dict else 0
        for sub_name, sub_t in timings.items():
            sub_m = int(mem_dict.get(sub_name, 0))
            sub_rows.append(
                {
                    "instance": row["instance"],
                    "n": int(row["n"]),
                    "algorithm": row["algorithm"],
                    "subroutine": sub_name,
                    "time_s": float(sub_t),
                    "share_pct": (100.0 * float(sub_t) / total_sub_t) if total_sub_t > 0 else 0.0,
                    "memory_bytes": sub_m,
                    "memory_share_pct": (100.0 * sub_m / total_sub_m) if total_sub_m > 0 else 0.0,
                }
            )
    sub_df = pd.DataFrame(sub_rows)
    sub_df.to_csv(results_dir / "subroutine_breakdown.csv", index=False)

    # 3. Complexidade de subrotinas críticas (ex: mst, min_weight_matching, greedy_insertion)
    if not sub_df.empty:
        for (alg, sub_name), grp in sub_df.groupby(["algorithm", "subroutine"]):
            if sub_name not in ("mst", "min_weight_matching", "greedy_insertion", "convex_hull"):
                continue
            n_vals = grp["n"].to_numpy(dtype=np.float64)
            for metric_col, target_name in [("time_s", "time_s"), ("memory_bytes", "memory_bytes")]:
                y_vals = grp[metric_col].to_numpy(dtype=np.float64)
                for fit in _fit_complexity_curves(n_vals, y_vals):
                    reg_rows.append(
                        {"algorithm": f"{alg}:{sub_name}", "target": target_name, **fit}
                    )

    # 4. Complexidade do Branch-and-Bound (exponencial para tempo/nós e polinomial para memória)
    if not bnb_df.empty:
        bnb_copy = bnb_df.copy()
        bnb_copy["is_timeout"] = _parse_bool_series(bnb_copy["timed_out"])
        bnb_copy["bnb_time_s"] = pd.to_numeric(bnb_copy["elapsed_time_s"], errors="coerce")
        for lb_mode, grp in bnb_copy.groupby("lb_mode"):
            # Memória em todas as instâncias por N
            mem_by_n = grp.groupby("n")["peak_memory_bytes"].max().reset_index()
            for fit in _fit_complexity_curves(
                mem_by_n["n"].to_numpy(dtype=np.float64),
                mem_by_n["peak_memory_bytes"].to_numpy(dtype=np.float64),
            ):
                reg_rows.append({"algorithm": f"bnb:{lb_mode}", "target": "memory_bytes", **fit})

            solved = grp[~grp["is_timeout"] & grp["bnb_time_s"].notna()]
            if len(solved) >= 2:
                agg_solved = (
                    solved.groupby("n")[["bnb_time_s", "nodes_expanded"]].median().reset_index()
                )
                n_s = agg_solved["n"].to_numpy(dtype=np.float64)
                for fit in _fit_exponential_curve(n_s, agg_solved["bnb_time_s"].to_numpy(dtype=np.float64)):
                    reg_rows.append({"algorithm": f"bnb:{lb_mode}", "target": "time_s", **fit})
                for fit in _fit_exponential_curve(
                    n_s, agg_solved["nodes_expanded"].to_numpy(dtype=np.float64)
                ):
                    reg_rows.append(
                        {"algorithm": f"bnb:{lb_mode}", "target": "nodes_expanded", **fit}
                    )

    reg_df = pd.DataFrame(reg_rows)
    reg_df.to_csv(results_dir / "complexity_regression.csv", index=False)

    # Gráfico 1: Tempo vs N com curvas de regressão ajustadas
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for alg, grp in approx_df.groupby("algorithm"):
        g_sorted = grp.sort_values("n")
        n_emp = g_sorted["n"].to_numpy(dtype=np.float64)
        y_emp = g_sorted["elapsed_time_s"].to_numpy(dtype=np.float64)
        (line,) = ax.plot(n_emp, y_emp, marker="o", linestyle="none", label=f"{alg} (empírico)")

        alg_fits = reg_df[(reg_df["algorithm"] == alg) & (reg_df["target"] == "time_s")]
        if not alg_fits.empty:
            best_fit = alg_fits.sort_values("r2", ascending=False).iloc[0]
            model_name = str(best_fit["model"])
            if model_name in _COMPLEXITY_BASES:
                n_grid = np.linspace(float(n_emp.min()), float(n_emp.max()), 100)
                y_pred = np.maximum(
                    1e-9,
                    float(best_fit["coef"]) * _COMPLEXITY_BASES[model_name](n_grid)
                    + float(best_fit["intercept"]),
                )
                ax.plot(
                    n_grid,
                    y_pred,
                    color=line.get_color(),
                    linestyle="--",
                    alpha=0.8,
                    label=f"{alg} fit {model_name} (R²={float(best_fit['r2']):.2f})",
                )

    ax.set_xlabel("Número de vértices (N)")
    ax.set_ylabel("Tempo de execução (s)")
    ax.set_title("Complexidade Temporal Empírica e Curvas Ajustadas")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(figures_dir / "complexity_time.png", dpi=150)
    plt.close(fig)

    # Gráfico 1B: Memória vs N com curvas de regressão ajustadas
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for alg, grp in approx_df.groupby("algorithm"):
        g_sorted = grp.sort_values("n")
        n_emp = g_sorted["n"].to_numpy(dtype=np.float64)
        y_emp = g_sorted["peak_memory_bytes"].to_numpy(dtype=np.float64)
        (line,) = ax.plot(n_emp, y_emp, marker="s", linestyle="-", alpha=0.85, label=str(alg))
    ax.set_xlabel("Número de vértices (N)")
    ax.set_ylabel("Pico de memória (bytes)")
    ax.set_title("Complexidade Espacial Empírica: Heurísticas e Aproximados")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(figures_dir / "complexity_memory.png", dpi=150)
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
    """Consolida impacto do warm-start na poda e no trade-off de tempo, segregado por lb_mode."""
    if bnb_df.empty:
        return pd.DataFrame()

    merged = bnb_df.merge(
        approx_df[["instance", "algorithm", "elapsed_time_s"]].rename(
            columns={"algorithm": "strategy", "elapsed_time_s": "heuristic_time_s"}
        ),
        on=["instance", "strategy"],
        how="left",
    )
    merged["timed_out"] = _parse_bool_series(merged["timed_out"])
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
    cold_totals = (
        merged[merged["strategy"] == "cold_start"]
        .set_index(["instance", "lb_mode"])["total_time_s"]
        .to_dict()
    )

    merged["nodes_vs_cold_ratio"] = [
        row["nodes_expanded"] / cold_nodes[(row["instance"], row["lb_mode"])]
        if (row["instance"], row["lb_mode"]) in cold_nodes
        and cold_nodes[(row["instance"], row["lb_mode"])] > 0
        else np.nan
        for _, row in merged.iterrows()
    ]
    merged["time_saved_vs_cold_s"] = [
        cold_totals[(row["instance"], row["lb_mode"])] - row["total_time_s"]
        if (row["instance"], row["lb_mode"]) in cold_totals
        and pd.notna(cold_totals[(row["instance"], row["lb_mode"])])
        and pd.notna(row["total_time_s"])
        else np.nan
        for _, row in merged.iterrows()
    ]

    merged.to_csv(results_dir / "warmstart_summary.csv", index=False)

    # Seleciona um lb_mode principal (preferindo 'mst') para não misturar modos diferentes no gráfico
    primary_mode = "mst" if "mst" in merged["lb_mode"].values else str(merged["lb_mode"].iloc[0])
    mode_df = merged[merged["lb_mode"] == primary_mode]

    # Gráfico 3: Nós expandidos e nós podados por estratégia (incluindo instâncias completas)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col, title in [
        (axes[0], "nodes_expanded", f"Nós Expandidos (LB={primary_mode})"),
        (axes[1], "nodes_pruned", f"Nós Podados (LB={primary_mode})"),
    ]:
        piv = mode_df.pivot_table(index="n", columns="strategy", values=col, aggfunc="mean")
        if not piv.empty:
            piv.plot(kind="bar", ax=ax)
            ax.set_xlabel("Número de vértices (N)")
            ax.set_ylabel("Contagem de nós (escala log)")
            if (piv.to_numpy() > 0).any():
                ax.set_yscale("log")
            ax.set_title(title)
            ax.grid(True, linestyle="--", alpha=0.4)
    axes[0].get_legend().remove() if axes[0].get_legend() else None
    axes[1].legend(title="Estratégia", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(figures_dir / "warmstart_nodes.png", dpi=150)
    plt.close(fig)

    # Gráfico 3B: Trade-off de Tempo Total (Heurística + B&B) nas instâncias resolvidas
    solved_mode = mode_df[~mode_df["timed_out"] & mode_df["total_time_s"].notna()]
    if not solved_mode.empty:
        piv_t = solved_mode.pivot_table(
            index="n", columns="strategy", values="total_time_s", aggfunc="mean"
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        piv_t.plot(kind="bar", ax=ax)
        ax.set_xlabel("Número de vértices (N)")
        ax.set_ylabel("Tempo total: Heurística + B&B (s, log)")
        if (piv_t.to_numpy() > 0).any():
            ax.set_yscale("log")
        ax.set_title(f"Trade-off de Tempo do Warm-Start (LB={primary_mode})")
        ax.legend(title="Estratégia", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(figures_dir / "warmstart_time_tradeoff.png", dpi=150)
        plt.close(fig)

    return merged


def _build_sensitivity_analysis(
    features_df: pd.DataFrame,
    approx_df: pd.DataFrame,
    bnb_df: pd.DataFrame,
    results_dir: Path,
    figures_dir: Path,
) -> pd.DataFrame:
    """Calcula gap de aproximação sobre ótimos exatos e avalia sensibilidade com LOOCV e Spearman."""
    ref_costs: Dict[str, Tuple[float, str]] = {}
    for _, row in features_df.iterrows():
        inst = str(row["instance"])
        opt = pd.to_numeric(row["optimal_cost"], errors="coerce")
        if pd.notna(opt) and float(opt) > 0:
            ref_costs[inst] = (float(opt), "tsplib_exact")
            continue

        if not bnb_df.empty:
            sub = bnb_df[bnb_df["instance"] == inst].copy()
            sub["is_timeout"] = _parse_bool_series(sub["timed_out"])
            exact_sub = sub[~sub["is_timeout"]]
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
    gap_df["gap_pct"] = np.where(
        gap_df["ref_cost"] > 0,
        100.0 * (gap_df["cost"] - gap_df["ref_cost"]) / gap_df["ref_cost"],
        0.0,
    )
    gap_df.to_csv(results_dir / "gap_by_instance.csv", index=False)

    # Filtra estritamente instâncias com ótimo exato para evitar viés de gap_pct = 0%
    exact_gap_df = gap_df[gap_df["ref_source"].isin(["tsplib_exact", "bnb_exact"])].copy()
    if len(exact_gap_df) < 3:
        exact_gap_df = gap_df.copy()

    target_algs = [
        "nearest_neighbor",
        "convex_hull_insertion",
        "twice_around_the_tree",
        "christofides",
    ]
    imp_rows: List[Dict[str, object]] = []

    for alg in target_algs:
        sub = exact_gap_df[exact_gap_df["algorithm"] == alg].dropna(
            subset=_FEATURE_COLS + ["gap_pct"]
        )
        if len(sub) < 3:
            continue

        x_df = sub[_FEATURE_COLS].replace([np.inf, -np.inf], np.nan).dropna()
        n_samples = len(x_df)
        if n_samples < 3:
            continue

        x_raw = x_df.to_numpy(dtype=np.float64)
        y_vec = sub.loc[x_df.index, "gap_pct"].to_numpy(dtype=np.float64)
        y_has_var = float(np.std(y_vec)) > 1e-12

        x_scaled = StandardScaler().fit_transform(x_raw)
        # Usa Ridge quando n_samples <= p + 2 para evitar sistema subdeterminado/overfitting
        p_feats = len(_FEATURE_COLS)
        if n_samples <= p_feats + 2:
            lin = Ridge(alpha=1.0).fit(x_scaled, y_vec)
        else:
            lin = LinearRegression().fit(x_scaled, y_vec)

        lin_preds = lin.predict(x_scaled)
        lin_r2 = float(r2_score(y_vec, lin_preds)) if y_has_var else 0.0
        if n_samples > p_feats + 1:
            adjusted_r2 = float(1.0 - (1.0 - lin_r2) * (n_samples - 1) / (n_samples - p_feats - 1))
        else:
            adjusted_r2 = float("nan")

        min_leaf = max(2, n_samples // 5) if n_samples >= 6 else 1
        tree = DecisionTreeRegressor(
            max_depth=2,
            min_samples_leaf=min_leaf,
            random_state=42,
        ).fit(x_raw, y_vec)
        tree_r2 = float(r2_score(y_vec, tree.predict(x_raw))) if y_has_var else 0.0

        # Leave-One-Out Cross-Validation (LOOCV) para avaliar generalização real
        if n_samples >= 4 and y_has_var:
            loo = LeaveOneOut()
            loo_preds = cross_val_predict(
                DecisionTreeRegressor(max_depth=2, min_samples_leaf=min_leaf, random_state=42),
                x_raw,
                y_vec,
                cv=loo,
            )
            tree_loocv_rmse = float(np.sqrt(mean_squared_error(y_vec, loo_preds)))
        else:
            tree_loocv_rmse = float("nan")

        for idx, feat_name in enumerate(_FEATURE_COLS):
            feat_vals = x_raw[:, idx]
            if y_has_var and float(np.std(feat_vals)) > 1e-12:
                sp_res = spearmanr(feat_vals, y_vec)
                pe_res = pearsonr(feat_vals, y_vec)
                spearman_rho = float(sp_res.statistic)
                spearman_p = float(sp_res.pvalue)
                pearson_r = float(pe_res.statistic)
            else:
                spearman_rho = 0.0
                spearman_p = 1.0
                pearson_r = 0.0

            imp_rows.append(
                {
                    "algorithm": alg,
                    "feature": feat_name,
                    "n_samples": n_samples,
                    "spearman_rho": spearman_rho,
                    "spearman_p": spearman_p,
                    "pearson_r": pearson_r,
                    "tree_importance": float(tree.feature_importances_[idx]),
                    "linear_std_coef": float(lin.coef_[idx]),
                    "linear_r2": lin_r2,
                    "linear_adj_r2": adjusted_r2,
                    "tree_r2": tree_r2,
                    "tree_loocv_rmse": tree_loocv_rmse,
                }
            )

    imp_df = pd.DataFrame(imp_rows)
    imp_df.to_csv(results_dir / "sensitivity_feature_importance.csv", index=False)

    if not imp_df.empty:
        piv = imp_df.pivot(index="feature", columns="algorithm", values="tree_importance").fillna(0.0)
        fig, ax = plt.subplots(figsize=(9, 5))
        piv.plot(kind="barh", ax=ax)
        ax.set_xlabel("Importância na Árvore de Decisão (max_depth=2)")
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

    _build_complexity_and_subroutines(approx_df, bnb_df, results_dir, figures_dir)
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
