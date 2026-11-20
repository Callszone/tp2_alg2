"""Testes para extração de features e orquestração experimental (runner)."""

import csv
import json
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


def test_extract_instance_features_collinear_is_finite() -> None:
    """Pontos colineares devem produzir aspect_ratio finito (nunca inf/NaN)."""
    coords = np.array([[0.0, 0.0], [5.0, 0.0], [12.0, 0.0], [20.0, 0.0]])
    dist = compute_euc2d_matrix(coords)
    feats = extract_instance_features(coords, dist)
    assert np.isfinite(feats["aspect_ratio"]) and feats["aspect_ratio"] >= 1.0


def test_extract_instance_features_tsplib_eil51_n12() -> None:
    """Valida extração de features em instância real da TSPLIB."""
    inst = load_tsplib_euc2d(Path("data/tsplib/eil51_n12.tsp"))
    feats = extract_instance_features(inst.coords, inst.dist_matrix)

    assert np.isfinite(feats["aspect_ratio"]) and feats["aspect_ratio"] >= 1.0
    assert 0.0 < feats["convex_hull_ratio"] <= 1.0
    assert 0.0 <= feats["spatial_entropy"] <= 1.0
    assert feats["mst_max_degree"] >= 2.0
    assert feats["nn_dist_cv"] >= 0.0


def test_runner_parallel_workers_checkpoint_and_subroutine_memory(tmp_path: Path) -> None:
    """Testa runner com workers=2, múltiplos lb_modes, memória por subrotina e checkpoint/resume."""
    run_experiments(
        data_dir=Path("data/tsplib"),
        output_dir=tmp_path,
        timeout_sec=2.0,
        workers=2,
        lb_modes=["two_min", "mst"],
        max_n=14,
        skip_bnb=False,
        repeats=3,
        resume=True,
    )

    feats_csv = tmp_path / "instances_features.csv"
    approx_csv = tmp_path / "approximations_results.csv"
    bnb_csv = tmp_path / "bnb_results.csv"
    assert feats_csv.exists() and approx_csv.exists() and bnb_csv.exists()

    with approx_csv.open(encoding="utf-8") as f:
        approx_rows = list(csv.DictReader(f))
    assert len(approx_rows) == 10  # 2 instâncias x 5 estratégias
    for row in approx_rows:
        sub_mem = json.loads(row["subroutine_memory_json"])
        assert isinstance(sub_mem, dict) and len(sub_mem) > 0

    with bnb_csv.open(encoding="utf-8") as f:
        bnb_rows = list(csv.DictReader(f))
    assert len(bnb_rows) == 20  # 2 instâncias x 2 lb_modes x 5 estratégias

    # Verifica ordenação determinística por (n, instance, lb_mode, strategy)
    ns = [int(r["n"]) for r in bnb_rows]
    assert ns == sorted(ns)

    # Reexecutar com resume=True não deve duplicar linhas
    run_experiments(
        data_dir=Path("data/tsplib"),
        output_dir=tmp_path,
        timeout_sec=2.0,
        workers=2,
        lb_modes=["two_min", "mst"],
        max_n=14,
        skip_bnb=False,
        repeats=2,
        resume=True,
    )
    with bnb_csv.open(encoding="utf-8") as f:
        bnb_rows_after = list(csv.DictReader(f))
    assert len(bnb_rows_after) == 20


def test_runner_timeout_na_and_skip_bnb(tmp_path: Path) -> None:
    """Valida skip_bnb=True e gravação de NA quando timeout estoura."""
    skip_dir = tmp_path / "skip"
    run_experiments(
        data_dir=Path("data/tsplib"),
        output_dir=skip_dir,
        max_n=12,
        skip_bnb=True,
        repeats=2,
    )
    assert (skip_dir / "instances_features.csv").exists()
    assert (skip_dir / "approximations_results.csv").exists()
    assert not (skip_dir / "bnb_results.csv").exists()

    to_dir = tmp_path / "timeout"
    run_experiments(
        data_dir=Path("data/tsplib"),
        output_dir=to_dir,
        timeout_sec=1e-7,
        workers=1,
        lb_modes=["two_min"],
        max_n=20,
        skip_bnb=False,
        repeats=2,
    )
    with (to_dir / "bnb_results.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert any(r["timed_out"] == "True" and r["elapsed_time_s"] == "NA" for r in rows)
