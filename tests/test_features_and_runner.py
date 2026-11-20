"""Testes para extração de features e orquestração experimental (runner)."""

import csv
from pathlib import Path
import numpy as np

from src.features import extract_instance_features
from src.runner import run_experiments
from src.tsplib_parser import compute_euc2d_matrix, load_tsplib_euc2d


def test_extract_instance_features_unit_square() -> None:
    """Valida métricas geométricas e topológicas em um quadrado 10x10."""
    coords = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    dist = compute_euc2d_matrix(coords)
    feats = extract_instance_features(coords, dist)

    assert feats["aspect_ratio"] == 1.0
    assert feats["convex_hull_ratio"] == 1.0
    assert 0.0 <= feats["spatial_entropy"] <= 1.0
    assert feats["mst_max_degree"] == 2.0
    assert feats["mst_leaf_ratio"] == 0.5
    assert feats["mst_total_cost"] == 30.0
    assert feats["nn_dist_cv"] == 0.0


def test_extract_instance_features_tsplib_eil51_n12() -> None:
    """Valida extração de features em instância real da TSPLIB."""
    inst = load_tsplib_euc2d(Path("data/tsplib/eil51_n12.tsp"))
    feats = extract_instance_features(inst.coords, inst.dist_matrix)

    assert feats["aspect_ratio"] >= 1.0
    assert 0.0 < feats["convex_hull_ratio"] <= 1.0
    assert 0.0 <= feats["spatial_entropy"] <= 1.0
    assert feats["mst_max_degree"] >= 2.0
    assert feats["nn_dist_cv"] >= 0.0


def test_runner_end_to_end_dry_run(tmp_path: Path) -> None:
    """Executa dry-run do runner com N<=14 e valida os 3 CSVs gerados."""
    run_experiments(
        data_dir=Path("data/tsplib"),
        output_dir=tmp_path,
        timeout_sec=2.0,
        workers=1,
        lb_modes=["mst"],
        max_n=14,
        skip_bnb=False,
    )

    feats_csv = tmp_path / "instances_features.csv"
    approx_csv = tmp_path / "approximations_results.csv"
    bnb_csv = tmp_path / "bnb_results.csv"
    assert feats_csv.exists()
    assert approx_csv.exists()
    assert bnb_csv.exists()

    with feats_csv.open(encoding="utf-8") as f:
        feat_rows = list(csv.DictReader(f))
    assert len(feat_rows) == 2  # eil51_n12 e berlin52_n14

    with approx_csv.open(encoding="utf-8") as f:
        approx_rows = list(csv.DictReader(f))
    assert len(approx_rows) == 10  # 2 instâncias x 5 estratégias

    with bnb_csv.open(encoding="utf-8") as f:
        bnb_rows = list(csv.DictReader(f))
    assert len(bnb_rows) == 10  # 2 instâncias x 1 lb_mode x 5 estratégias
