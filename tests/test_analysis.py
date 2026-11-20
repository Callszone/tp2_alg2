"""Testes de integração para o pipeline de análise estatística e gráficos."""

from pathlib import Path
import pandas as pd

from src.analysis import run_analysis
from src.runner import run_experiments


def test_analysis_end_to_end_generates_tables_and_figures(tmp_path: Path) -> None:
    """Roda runner (N<=16) + analysis e valida todos os CSVs e PNGs gerados."""
    results_dir = tmp_path / "results"
    figures_dir = tmp_path / "figures"

    run_experiments(
        data_dir=Path("data/tsplib"),
        output_dir=results_dir,
        timeout_sec=1.0,
        workers=1,
        lb_modes=["mst"],
        max_n=16,
        skip_bnb=False,
    )
    run_analysis(results_dir=results_dir, figures_dir=figures_dir)

    expected_csvs = [
        "complexity_regression.csv",
        "subroutine_breakdown.csv",
        "warmstart_summary.csv",
        "gap_by_instance.csv",
        "sensitivity_feature_importance.csv",
    ]
    for name in expected_csvs:
        path = results_dir / name
        assert path.exists(), f"CSV ausente: {name}"
        assert len(pd.read_csv(path)) > 0

    expected_figs = [
        "complexity_time.png",
        "christofides_subroutines.png",
        "warmstart_nodes.png",
        "sensitivity_importance.png",
    ]
    for name in expected_figs:
        path = figures_dir / name
        assert path.exists(), f"Figura ausente: {name}"
        assert path.stat().st_size > 0
