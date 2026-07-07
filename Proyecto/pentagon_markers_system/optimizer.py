"""
optimizer.py
============

Explora el espacio de soluciones (2^20 = 1,048,576 combinaciones posibles)
de forma eficiente, sin enumeración exhaustiva, para encontrar marcadores
de alta calidad según evaluator.score().

Este módulo NO conoce los detalles internos de qué hace bueno a un
marcador (eso es responsabilidad de evaluator.py); solo sabe cómo buscar,
aceptar/rechazar soluciones y mantener diversidad entre ellas.

Estrategia
----------
- Simulated annealing con enfriamiento geométrico.
- Mutación: se invierten entre 1 y 3 bits aleatorios por iteración.
- Aceptación: si mejora el score se acepta siempre; si empeora, se acepta
  con probabilidad exp(-delta / T).
- Diversidad: al construir una población de marcadores, se descartan
  soluciones demasiado similares (distancia Hamming mínima) a las ya
  seleccionadas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

from geometry import PentagonMesh
from marker import Marker, N_BITS
from evaluator import score as default_score_fn, EvaluatorWeights


# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #
@dataclass
class AnnealingConfig:
    n_iterations: int = 3000
    t_initial: float = 1.0
    t_final: float = 0.01
    min_flips: int = 1
    max_flips: int = 3
    seed: Optional[int] = None


# --------------------------------------------------------------------------- #
# Optimización de un único marcador
# --------------------------------------------------------------------------- #
def optimize_single(geometry: PentagonMesh,
                     existing_markers: Optional[Sequence[Marker]] = None,
                     config: Optional[AnnealingConfig] = None,
                     weights: Optional[EvaluatorWeights] = None,
                     score_fn: Optional[Callable] = None,
                     initial: Optional[Marker] = None) -> Marker:
    """
    Ejecuta simulated annealing para encontrar un marcador de alta calidad.

    Parameters
    ----------
    geometry : PentagonMesh
        Estructura geométrica fija.
    existing_markers : list[Marker], opcional
        Marcadores ya seleccionados (para separabilidad).
    config : AnnealingConfig, opcional
        Parámetros del algoritmo de recocido simulado.
    weights : EvaluatorWeights, opcional
        Pesos de las métricas de evaluación.
    score_fn : callable, opcional
        Función de score alternativa, con firma compatible con
        evaluator.score(marker, geometry, existing_markers, weights).
    initial : Marker, opcional
        Punto de partida. Si no se entrega, se genera un marcador aleatorio válido.

    Returns
    -------
    Marker
        El mejor marcador encontrado durante la búsqueda (con su score asignado).
    """
    config = config or AnnealingConfig()
    score_fn = score_fn or default_score_fn
    rng = np.random.default_rng(config.seed)

    # --- Inicialización ---
    current = initial.copy() if initial is not None else _random_valid_marker(rng)
    current_score = score_fn(current, geometry, existing_markers, weights)

    best = current.copy()
    best.score = current_score
    best_score = current_score

    # Enfriamiento geométrico: T_i = T0 * (Tf/T0)^(i / n_iterations)
    if config.n_iterations <= 1:
        temperatures = [config.t_initial]
    else:
        ratio = config.t_final / config.t_initial
        temperatures = [
            config.t_initial * (ratio ** (i / (config.n_iterations - 1)))
            for i in range(config.n_iterations)
        ]

    for T in temperatures:
        candidate = current.copy()
        n_flips = int(rng.integers(config.min_flips, config.max_flips + 1))
        flip_indices = rng.choice(N_BITS, size=n_flips, replace=False)
        candidate.flip_many(flip_indices)

        if not candidate.validate():
            continue  # descartar candidatos triviales (todo-ceros/todo-unos, etc.)

        candidate_score = score_fn(candidate, geometry, existing_markers, weights)
        delta = candidate_score - current_score  # positivo = mejora

        accept = False
        if delta >= 0:
            accept = True
        else:
            prob = math.exp(delta / max(T, 1e-9))
            accept = rng.random() < prob

        if accept:
            current = candidate
            current_score = candidate_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score

    best.score = best_score
    return best


def _random_valid_marker(rng: np.random.Generator, max_attempts: int = 100) -> Marker:
    """Genera un marcador aleatorio que cumpla validate(); reintenta si es necesario."""
    for _ in range(max_attempts):
        m = Marker.random(rng)
        if m.validate():
            return m
    # como último recurso, forzar balance exacto 10/10
    bits = np.array([1] * (N_BITS // 2) + [0] * (N_BITS // 2), dtype=np.uint8)
    rng.shuffle(bits)
    return Marker(bits=bits)


# --------------------------------------------------------------------------- #
# Generación de población
# --------------------------------------------------------------------------- #
def generate_population(geometry: PentagonMesh,
                         n: int,
                         config: Optional[AnnealingConfig] = None,
                         weights: Optional[EvaluatorWeights] = None,
                         score_fn: Optional[Callable] = None,
                         enforce_diversity: bool = True,
                         min_hamming: int = 6,
                         seed: Optional[int] = None) -> List[Marker]:
    """
    Genera una población de n marcadores optimizados, opcionalmente
    exigiendo diversidad incremental entre ellos (cada nuevo marcador se
    optimiza teniendo en cuenta los ya seleccionados como `existing_markers`,
    lo que penaliza baja separabilidad).

    Parameters
    ----------
    geometry : PentagonMesh
    n : int
        Número de marcadores a generar.
    config : AnnealingConfig, opcional
    weights : EvaluatorWeights, opcional
    score_fn : callable, opcional
    enforce_diversity : bool
        Si es True, cada marcador se optimiza considerando los anteriores
        como `existing_markers` (mejora separabilidad).
    min_hamming : int
        Distancia Hamming mínima aceptable respecto a los ya seleccionados;
        si un candidato no la cumple, se reintenta con una nueva semilla.
    seed : int, opcional
        Semilla base; cada marcador usa una semilla derivada distinta.

    Returns
    -------
    list[Marker]
    """
    base_config = config or AnnealingConfig()
    population: List[Marker] = []
    rng_master = np.random.default_rng(seed)

    attempt = 0
    max_total_attempts = n * 20  # margen de seguridad

    while len(population) < n and attempt < max_total_attempts:
        attempt += 1
        run_seed = int(rng_master.integers(0, 2**31 - 1))
        run_config = AnnealingConfig(
            n_iterations=base_config.n_iterations,
            t_initial=base_config.t_initial,
            t_final=base_config.t_final,
            min_flips=base_config.min_flips,
            max_flips=base_config.max_flips,
            seed=run_seed,
        )

        existing = population if enforce_diversity else None
        candidate = optimize_single(
            geometry, existing_markers=existing, config=run_config,
            weights=weights, score_fn=score_fn,
        )

        if enforce_diversity and population:
            min_dist = min(candidate.hamming(m) for m in population)
            if min_dist < min_hamming:
                continue  # demasiado similar a uno existente, reintentar

        population.append(candidate)

    return population


# --------------------------------------------------------------------------- #
# Selección de subconjunto diverso
# --------------------------------------------------------------------------- #
def select_diverse(markers: Sequence[Marker], k: int = 12,
                    min_hamming: int = 0) -> List[Marker]:
    """
    Selecciona un subconjunto de k marcadores maximizando la diversidad
    (distancia Hamming mutua), usando una heurística greedy:

        1. Ordenar candidatos por score descendente.
        2. Tomar el de mayor score como semilla.
        3. Iterativamente, agregar el candidato restante que maximice la
           distancia Hamming mínima respecto al conjunto ya seleccionado.

    Parameters
    ----------
    markers : list[Marker]
        Candidatos (deben tener `score` asignado, o se asumirá 0).
    k : int
        Número de marcadores a seleccionar.
    min_hamming : int
        Si > 0, se descartan candidatos cuya distancia mínima al conjunto
        seleccionado sea menor a este umbral (pueden quedar menos de k).

    Returns
    -------
    list[Marker]
    """
    if not markers:
        return []

    pool = sorted(markers, key=lambda m: (m.score if m.score is not None else 0.0),
                  reverse=True)

    selected: List[Marker] = [pool[0]]
    remaining = pool[1:]

    while len(selected) < k and remaining:
        best_candidate = None
        best_min_dist = -1

        for cand in remaining:
            min_dist = min(cand.hamming(s) for s in selected)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_candidate = cand

        if best_candidate is None:
            break
        if min_hamming > 0 and best_min_dist < min_hamming:
            break  # ya no hay candidatos suficientemente distintos

        selected.append(best_candidate)
        remaining.remove(best_candidate)

    return selected
