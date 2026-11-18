"""Testes para o parser TSPLIB EUC_2D e o conjunto de instâncias selecionado."""

from pathlib import Path
import numpy as np

from src.branch_and_bound import solve_branch_and_bound
from src.tsplib_parser import KNOWN_OPTIMA, list_tsplib_instances, load_tsplib_euc2d
from tests.test_contracts import compute_tour_cost


def test_list_tsplib_instances_dataset_composition() -> None:
    """Valida as 15 instâncias EUC_2D (80% N<=40, 20% N in [45, 75])."""
    instances = list_tsplib_instances(Path("data/tsplib"))
    assert len(instances) == 15

    main_band = [inst for inst in instances if inst.dimension <= 40]
    limit_band = [inst for inst in instances if 45 <= inst.dimension <= 75]
    assert len(main_band) == 12
    assert len(limit_band) == 3

    for inst in instances:
        assert inst.edge_weight_type == "EUC_2D"
        assert inst.coords.shape == (inst.dimension, 2)
        assert inst.dist_matrix.shape == (inst.dimension, inst.dimension)
        assert np.all(np.diag(inst.dist_matrix) == 0.0)
        assert np.allclose(inst.dist_matrix, inst.dist_matrix.T)
        if inst.name in KNOWN_OPTIMA:
            assert inst.optimal_cost == KNOWN_OPTIMA[inst.name]


def test_load_tsplib_rejects_non_euc2d(tmp_path: Path) -> None:
    """Parser deve rejeitar arquivos cujo EDGE_WEIGHT_TYPE não seja EUC_2D."""
    bad_file = tmp_path / "bad_geo.tsp"
    bad_file.write_text(
        "NAME : bad_geo\nTYPE : TSP\nDIMENSION : 2\nEDGE_WEIGHT_TYPE : GEO\n"
        "NODE_COORD_SECTION\n1 0.0 0.0\n2 1.0 1.0\nEOF\n",
        encoding="utf-8",
    )
    try:
        load_tsplib_euc2d(bad_file)
        assert False, "Esperava ValueError para EDGE_WEIGHT_TYPE != EUC_2D"
    except ValueError:
        pass


def test_bnb_solves_tsplib_subinstance_eil51_n12() -> None:
    """Integração: parser TSPLIB + Branch-and-Bound (MST) em eil51_n12."""
    inst = load_tsplib_euc2d(Path("data/tsplib/eil51_n12.tsp"))
    res = solve_branch_and_bound(inst.dist_matrix, lower_bound_mode="mst", timeout_sec=10.0)
    assert not res.timed_out
    assert abs(compute_tour_cost(res.best_tour, inst.dist_matrix) - res.best_cost) < 1e-9
