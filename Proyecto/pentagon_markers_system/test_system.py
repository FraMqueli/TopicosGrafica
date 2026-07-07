"""
test_system.py
===============

Pruebas básicas de humo (smoke tests) para validar la integridad del
sistema completo. No pretende ser una suite exhaustiva, sino verificar
que cada módulo cumple su contrato mínimo.

Ejecutar con:
    python -m pytest test_system.py -v
o simplemente:
    python test_system.py
"""

import numpy as np

from geometry import PentagonMesh
from marker import Marker, N_BITS
from evaluator import score, EvaluatorWeights
from optimizer import optimize_single, generate_population, select_diverse, AnnealingConfig


def test_geometry_structure():
    geo = PentagonMesh()
    triangles = geo.get_triangles()
    assert len(triangles) == 20
    graph = geo.get_graph()
    assert len(graph) == 20
    for tid, neighbors in graph.items():
        assert 2 <= len(neighbors) <= 3, f"triángulo {tid} tiene {len(neighbors)} vecinos"
    # el grafo debe ser simétrico
    for tid, neighbors in graph.items():
        for n in neighbors:
            assert tid in graph[n], "adyacencia no simétrica"


def test_automorphisms_are_permutations():
    geo = PentagonMesh()
    autos = geo.get_automorphisms()
    assert len(autos) == 10  # D5: 5 rotaciones + 5 reflexiones
    for perm in autos:
        assert sorted(perm.tolist()) == list(range(20))
    # la identidad debe estar presente
    assert any(np.array_equal(p, np.arange(20)) for p in autos)


def test_marker_basic_ops():
    rng = np.random.default_rng(0)
    m1 = Marker.random(rng)
    assert m1.bits.shape == (N_BITS,)
    m2 = m1.copy()
    assert m1 == m2
    m2.flip(0)
    assert m1 != m2
    assert m1.hamming(m2) == 1
    assert Marker.from_int(m1.to_int()) == m1


def test_marker_validate_rejects_extremes():
    all_zero = Marker(bits=np.zeros(N_BITS, dtype=np.uint8))
    all_one = Marker(bits=np.ones(N_BITS, dtype=np.uint8))
    assert not all_zero.validate()
    assert not all_one.validate()


def test_score_in_valid_range():
    geo = PentagonMesh()
    rng = np.random.default_rng(1)
    for _ in range(20):
        m = Marker.random(rng)
        s = score(m, geo)
        assert 0.0 <= s <= 1.0 + 1e-9


def test_score_penalizes_symmetric_marker():
    """Un marcador totalmente simétrico (invariante bajo rotación) debe
    tener symmetry_score y orientation_score muy bajos."""
    geo = PentagonMesh()
    # patrón repetido idéntico en cada una de las 5 cuñas -> invariante a rotación
    wedge_pattern = [1, 0, 1, 0]
    bits = np.array(wedge_pattern * 5, dtype=np.uint8)
    m = Marker(bits=bits)
    s, breakdown = score(m, geo, return_breakdown=True)
    assert breakdown["symmetry"] < 0.2
    assert breakdown["orientation"] < 0.2


def test_optimizer_improves_over_random():
    geo = PentagonMesh()
    rng = np.random.default_rng(2)
    random_scores = [score(Marker.random(rng), geo) for _ in range(30)]
    avg_random = sum(random_scores) / len(random_scores)

    optimized = optimize_single(geo, config=AnnealingConfig(n_iterations=1000, seed=42))
    assert optimized.score >= avg_random, (
        f"optimizado ({optimized.score:.4f}) no superó promedio aleatorio ({avg_random:.4f})"
    )


def test_generate_population_diversity():
    geo = PentagonMesh()
    pop = generate_population(
        geo, n=5, config=AnnealingConfig(n_iterations=300), min_hamming=4, seed=3,
    )
    assert len(pop) == 5
    for i in range(len(pop)):
        for j in range(i + 1, len(pop)):
            assert pop[i].hamming(pop[j]) >= 4


def test_select_diverse_returns_k():
    geo = PentagonMesh()
    rng = np.random.default_rng(4)
    candidates = [Marker.random(rng) for _ in range(20)]
    for c in candidates:
        score(c, geo)
    selected = select_diverse(candidates, k=6)
    assert len(selected) == 6


def run_all():
    tests = [
        test_geometry_structure,
        test_automorphisms_are_permutations,
        test_marker_basic_ops,
        test_marker_validate_rejects_extremes,
        test_score_in_valid_range,
        test_score_penalizes_symmetric_marker,
        test_optimizer_improves_over_random,
        test_generate_population_diversity,
        test_select_diverse_returns_k,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"OK   {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests pasaron")


if __name__ == "__main__":
    run_all()
