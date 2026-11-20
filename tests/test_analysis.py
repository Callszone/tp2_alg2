"""Testes unitários e de integração para o pipeline de análise estatística e gráficos."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.analysis import _fit_complexity_curves, run_analysis
from src.runner import run_experiments


def test_fit_complexity_curves_recovers_quadratic_and_non_negative_coef() -> None:
    """Valida que _fit_complexity_curves identifica O(N^2) com R2 > 0.999 e coef >= 0."""
    n_vals = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    y_vals = 2.5 * (n_vals**2) + 10.0
    fits = {f["model"]: f for f in _fit_complexity_curves(n_vals, y_vals)}

    assert fits["O(N^2)"]["r2"] == pytest.approx(1.0, abs=1e-6)
    assert fits["O(N^2)"]["coef"] == pytest.approx(2.5, rel=1e-4)
    assert fits["O(N^2)"]["rmse"] < fits["O(N)"]["rmse"]
    for fit in fits.values():
        assert float(fit["coef"]) >= 0.0


def test_analysis_end_to_end_statistical_properties_and_figures(tmp_path: Path) -> None:
    """Roda runner (N<=18, two_min e mst) + analysis e valida propriedades numéricas dos CSVs e PNGs."""
    results_dir = tmp_path / "results"
    figures_dir = tmp_path / "figures"

    run_experiments(
        data_dir=Path("data/tsplib"),
        output_dir=results_dir,
        timeout_sec=1.5,
        workers=2,
        lb_modes=["two_min", "mst"],
        max_n=18,
        skip_bnb=False,
        repeats=3,
    )
    run_analysis(results_dir=results_dir, figures_dir=figures_dir)

    reg_df = pd.read_csv(results_dir / "complexity_regression.csv")
    sub_df = pd.read_csv(results_dir / "subroutine_breakdown.csv")
    warm_df = pd.read_csv(results_dir / "warmstart_summary.csv")
    gap_df = pd.read_csv(results_dir / "gap_by_instance.csv")
    imp_df = pd.read_csv(results_dir / "sensitivity_feature_importance.csv")

    # 1. Regressão de complexidade inclui algoritmos, subrotinas e B&B com coef >= 0
    assert (reg_df["coef"] >= -1e-12).all()
    assert any(reg_df["algorithm"].str.startswith("christofides:"))
    assert any(reg_df["algorithm"].str.startswith("bnb:"))

    # 2. Subroutine breakdown soma 100% de share_pct por (instance, algorithm) e possui memória
    grouped_shares = sub_df.groupby(["instance", "algorithm"])["share_pct"].sum()
    for val in grouped_shares:
        assert val == pytest.approx(100.0, abs=1e-3)
    assert "memory_bytes" in sub_df.columns
    assert (sub_df["memory_bytes"] >= 0).all()

    # 3. Warm-start summary: prune_ratio em [0, 1] e cold_start tem nodes_vs_cold_ratio == 1.0 por lb_mode
    assert ((warm_df["prune_ratio"] >= 0.0) & (warm_df["prune_ratio"] <= 1.0)).all()
    cold_rows = warm_df[warm_df["strategy"] == "cold_start"]
    assert np.allclose(cold_rows["nodes_vs_cold_ratio"], 1.0)

    # 4. Gap e Sensibilidade: todas as referências são exatas (tsplib_exact) e colunas de Spearman/LOOCV existem
    assert (gap_df["ref_source"] == "tsplib_exact").all()
    assert (gap_df["gap_pct"] >= -1e-9).all()
    for col in ("spearman_rho", "spearman_p", "pearson_r", "tree_loocv_rmse"):
        assert col in imp_df.columns

    expected_figs = [
        "complexity_time.png",
        "complexity_memory.png",
        "christofides_subroutines.png",
        "warmstart_nodes.png",
        "warmstart_time_tradeoff.png",
        "sensitivity_importance.png",
    ]
    for name in expected_figs:
        path = figures_dir / name
        assert path.exists(), f"Figura ausente: {name}"
        assert path.stat().st_size > 0
