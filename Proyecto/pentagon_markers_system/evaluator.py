"""
evaluator.py
============

Evalúa la calidad de un marcador en función de su utilidad para
Structure-from-Motion (COLMAP), sin modificar el pipeline de reconstrucción.

Este módulo NO conoce cómo se generan o buscan los marcadores (eso es
responsabilidad de optimizer.py); solo sabe medir qué tan "bueno" es un
marcador dado.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from geometry import PentagonMesh
from marker import Marker, N_BITS


# --------------------------------------------------------------------------- #
# Pesos por defecto
# --------------------------------------------------------------------------- #
@dataclass
class EvaluatorWeights:
    balance: float = 1.0
    transitions: float = 1.5
    symmetry: float = 2.0
    connectivity: float = 1.5
    orientation: float = 2.0
    separability: float = 2.5


# --------------------------------------------------------------------------- #
# Métricas individuales
# --------------------------------------------------------------------------- #
def balance_score(marker: Marker, target_ratio: float = 0.5) -> float:
    """
    Retorna un score en [0, 1]; 1.0 cuando la proporción de unos es exactamente
    target_ratio, decreciendo linealmente hacia 0 en los extremos (todo-ceros
    o todo-unos).
    """
    n_ones = int(marker.bits.sum())
    ratio = n_ones / N_BITS
    return float(1.0 - abs(ratio - target_ratio) / max(target_ratio, 1 - target_ratio))


def transitions_score(marker: Marker, graph: Dict[int, List[int]]) -> float:
    """
    Retorna la fracción de aristas del grafo de adyacencia cuyos dos
    triángulos tienen bits distintos (proxy de riqueza de textura / densidad
    de features detectables). Rango [0, 1].
    """
    bits = marker.bits
    total_edges = 0
    diff_edges = 0
    for tid, neighbors in graph.items():
        for nid in neighbors:
            if nid > tid:  # contar cada arista una sola vez
                total_edges += 1
                if bits[tid] != bits[nid]:
                    diff_edges += 1
    if total_edges == 0:
        return 0.0
    return diff_edges / total_edges


def symmetry_score(marker: Marker, automorphisms: Sequence[np.ndarray]) -> float:
    """
    Retorna un score en [0, 1]; 1.0 significa que el marcador NO es invariante
    bajo ninguna simetría no trivial del pentágono (rotaciones/reflexiones D5),
    lo cual es deseable para evitar ambigüedad de pose.

    Se penaliza según qué tan cerca (en distancia de Hamming normalizada)
    está el marcador de ser invariante bajo cada automorfismo no trivial.
    """
    bits = marker.bits
    worst_similarity = 0.0  # 1.0 = idéntico bajo esa simetría (malo)

    for perm in automorphisms:
        if np.array_equal(perm, np.arange(N_BITS)):
            continue  # identidad, se ignora
        transformed = bits[perm]
        hamming = int(np.sum(bits != transformed))
        similarity = 1.0 - hamming / N_BITS  # 1.0 = totalmente simétrico
        worst_similarity = max(worst_similarity, similarity)

    return float(1.0 - worst_similarity)


def connectivity_score(marker: Marker, graph: Dict[int, List[int]]) -> float:
    """
    Penaliza la presencia de regiones homogéneas grandes (blobs de bits
    iguales conectados). Se calcula el tamaño de la componente conexa más
    grande (BFS) entre triángulos con el mismo bit, y se retorna
    1 - (tamaño_max_componente / N_BITS), de forma que valores altos
    indican buena fragmentación espacial.
    """
    bits = marker.bits
    visited = [False] * N_BITS
    max_component = 0

    for start in range(N_BITS):
        if visited[start]:
            continue
        # BFS
        queue = [start]
        visited[start] = True
        size = 0
        value = bits[start]
        while queue:
            node = queue.pop()
            size += 1
            for neighbor in graph.get(node, []):
                if not visited[neighbor] and bits[neighbor] == value:
                    visited[neighbor] = True
                    queue.append(neighbor)
        max_component = max(max_component, size)

    return float(1.0 - (max_component / N_BITS))


def orientation_score(marker: Marker, automorphisms: Sequence[np.ndarray],
                       n_rotations: int = 5) -> float:
    """
    Mide si existe un patrón único que permita distinguir la orientación del
    marcador: retorna la distancia de Hamming mínima (normalizada) entre el
    marcador y CADA una de sus rotaciones no triviales (se asume que las
    primeras n_rotations entradas de `automorphisms`, tal como las entrega
    PentagonMesh.get_automorphisms(), corresponden a rotaciones).

    Un score alto (cercano a 1) indica que ninguna rotación del marcador
    coincide con el marcador original, por lo tanto su orientación es
    inequívoca al ser observado.
    """
    bits = marker.bits
    rotations = automorphisms[:n_rotations]
    min_hamming = N_BITS  # peor caso inicial

    for perm in rotations:
        if np.array_equal(perm, np.arange(N_BITS)):
            continue
        transformed = bits[perm]
        hamming = int(np.sum(bits != transformed))
        min_hamming = min(min_hamming, hamming)

    if min_hamming == N_BITS:
        # no había rotaciones no triviales (caso degenerado)
        return 1.0
    return float(min_hamming / N_BITS)


def separability_score(marker: Marker, existing_markers: Optional[Sequence[Marker]],
                        target_min_distance: int = 6) -> float:
    """
    Garantiza que los marcadores sean distinguibles entre sí: retorna un
    score en [0, 1] basado en la distancia Hamming mínima respecto al
    conjunto de marcadores ya seleccionados. Si no hay marcadores previos,
    retorna 1.0 (no hay conflicto posible).
    """
    if not existing_markers:
        return 1.0

    min_dist = min(marker.hamming(other) for other in existing_markers)
    return float(min(1.0, min_dist / target_min_distance))


# --------------------------------------------------------------------------- #
# Función principal
# --------------------------------------------------------------------------- #
def score(marker: Marker,
          geometry: PentagonMesh,
          existing_markers: Optional[Sequence[Marker]] = None,
          weights: Optional[EvaluatorWeights] = None,
          return_breakdown: bool = False):
    """
    Calcula el score total de calidad de un marcador.

    Parameters
    ----------
    marker : Marker
        Marcador a evaluar.
    geometry : PentagonMesh
        Estructura geométrica fija (provee grafo y automorfismos).
    existing_markers : list[Marker], opcional
        Marcadores ya seleccionados, usados para separabilidad.
    weights : EvaluatorWeights, opcional
        Pesos de cada métrica. Si no se entrega, se usan los valores por defecto.
    return_breakdown : bool
        Si es True, retorna también un diccionario con cada métrica individual.

    Returns
    -------
    float, o (float, dict) si return_breakdown=True
    """
    weights = weights or EvaluatorWeights()
    graph = geometry.get_graph()
    automorphisms = geometry.get_automorphisms()

    metrics = {
        "balance": balance_score(marker),
        "transitions": transitions_score(marker, graph),
        "symmetry": symmetry_score(marker, automorphisms),
        "connectivity": connectivity_score(marker, graph),
        "orientation": orientation_score(marker, automorphisms),
        "separability": separability_score(marker, existing_markers),
    }

    total_weight = (
        weights.balance + weights.transitions + weights.symmetry +
        weights.connectivity + weights.orientation + weights.separability
    )

    total = (
        weights.balance * metrics["balance"] +
        weights.transitions * metrics["transitions"] +
        weights.symmetry * metrics["symmetry"] +
        weights.connectivity * metrics["connectivity"] +
        weights.orientation * metrics["orientation"] +
        weights.separability * metrics["separability"]
    ) / total_weight

    marker.score = float(total)

    if return_breakdown:
        return float(total), metrics
    return float(total)
