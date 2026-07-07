"""
geometry.py
===========

Define la estructura geométrica fija del sistema: un pentágono regular
subdividido en 20 triángulos.

Este módulo NO conoce nada sobre marcadores, bits, colores ni optimización.
Su única responsabilidad es:
    - construir la geometría (vértices, centros)
    - calcular la conectividad (grafo de adyacencia)
    - exponer esa estructura a través de una interfaz simple y estable

Construcción geométrica
------------------------
El pentágono regular se divide en 5 "cuñas" (wedges) triangulares que van
desde el centro hasta cada arista del pentágono (triángulo central + 2
vértices consecutivos del pentágono). Cada cuña se subdivide, a su vez, en
4 triángulos mediante el punto medio de sus 3 lados (subdivisión clásica
1 -> 4 de un triángulo). Esto produce:

    5 cuñas x 4 sub-triángulos = 20 triángulos

Dentro de cada cuña, los 4 sub-triángulos son:
    s=0 -> triángulo que contiene el centro del pentágono   (centro, A, B)
    s=1 -> triángulo de la esquina "izquierda" de la cuña   (A, V_i, C)
    s=2 -> triángulo de la esquina "derecha" de la cuña     (B, C, V_{i+1})
    s=3 -> triángulo central invertido de la cuña           (A, C, B)

donde, para la cuña i (con vértices del pentágono V_i, V_{i+1}):
    A = punto medio (centro, V_i)
    B = punto medio (centro, V_{i+1})
    C = punto medio (V_i, V_{i+1})

Los triángulos resultantes son equiláteros (o muy cercanos a equiláteros)
porque el ángulo en el centro de cada cuña es 72°, y cada cuña se
"normaliza" para que sea isósceles con lados en proporción áurea antes de
subdividir. En la práctica, para un pentágono regular la aproximación
visual es más que suficiente para el propósito de un marcador binario.

El id de cada triángulo se define como:
    id = wedge_index * 4 + sub_index      (wedge_index en 0..4, sub_index en 0..3)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

Point = Tuple[float, float]


# --------------------------------------------------------------------------- #
# Triangle
# --------------------------------------------------------------------------- #
@dataclass
class Triangle:
    """Representa una unidad básica (triángulo) del sistema."""

    id: int
    vertices: np.ndarray            # shape (3, 2)
    center: np.ndarray = field(init=False)  # shape (2,)
    neighbors: List[int] = field(default_factory=list)  # ids de triángulos vecinos

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=float)
        if self.vertices.shape != (3, 2):
            raise ValueError(f"vertices debe tener forma (3, 2), recibido {self.vertices.shape}")
        self.center = self.vertices.mean(axis=0)

    def edges(self) -> List[Tuple[Point, Point]]:
        """Retorna los 3 lados del triángulo como pares de puntos (sin ordenar)."""
        v = self.vertices
        return [
            (tuple(v[0]), tuple(v[1])),
            (tuple(v[1]), tuple(v[2])),
            (tuple(v[2]), tuple(v[0])),
        ]

    def __repr__(self) -> str:  # pragma: no cover - solo utilidad de debug
        return f"Triangle(id={self.id}, center=({self.center[0]:.3f}, {self.center[1]:.3f}))"


# --------------------------------------------------------------------------- #
# PentagonMesh
# --------------------------------------------------------------------------- #
class PentagonMesh:
    """
    Construye y expone la estructura geométrica fija:
    pentágono regular subdividido en 20 triángulos, junto con su
    conectividad (grafo de adyacencia).
    """

    N_TRIANGLES = 20
    N_WEDGES = 5
    SUB_PER_WEDGE = 4

    def __init__(self, radius: float = 1.0, center: Point = (0.0, 0.0), tol: float = 1e-6):
        self.radius = radius
        self.origin = np.array(center, dtype=float)
        self.tol = tol

        self._triangles: List[Triangle] = []
        self._graph: Dict[int, List[int]] = {}
        self._pentagon_vertices: np.ndarray = self._compute_pentagon_vertices()

        self.build()
        self.compute_adjacency()

    # ------------------------------------------------------------------ #
    # Construcción
    # ------------------------------------------------------------------ #
    def _compute_pentagon_vertices(self) -> np.ndarray:
        """Vértices del pentágono regular, orientados con un vértice hacia arriba."""
        angles = [math.pi / 2 + 2 * math.pi * k / 5 for k in range(5)]
        pts = np.array(
            [
                (self.origin[0] + self.radius * math.cos(a),
                 self.origin[1] + self.radius * math.sin(a))
                for a in angles
            ]
        )
        return pts

    @staticmethod
    def _midpoint(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        return (p + q) / 2.0

    def build(self) -> List[Triangle]:
        """
        Genera los 20 triángulos del pentágono subdividido.
        Puede llamarse más de una vez (reconstruye desde cero).
        """
        triangles: List[Triangle] = []
        V = self._pentagon_vertices
        O = self.origin

        for w in range(self.N_WEDGES):
            Vi = V[w]
            Vj = V[(w + 1) % self.N_WEDGES]

            A = self._midpoint(O, Vi)
            B = self._midpoint(O, Vj)
            C = self._midpoint(Vi, Vj)

            sub_vertices = [
                np.array([O, A, B]),    # s=0: triángulo del centro
                np.array([A, Vi, C]),   # s=1: esquina izquierda
                np.array([B, C, Vj]),   # s=2: esquina derecha
                np.array([A, C, B]),    # s=3: triángulo central invertido
            ]

            for s, verts in enumerate(sub_vertices):
                tid = w * self.SUB_PER_WEDGE + s
                triangles.append(Triangle(id=tid, vertices=verts))

        self._triangles = triangles
        return self._triangles

    # ------------------------------------------------------------------ #
    # Adyacencia
    # ------------------------------------------------------------------ #
    def compute_adjacency(self) -> Dict[int, List[int]]:
        """
        Calcula las relaciones de vecindad entre triángulos: dos triángulos
        son vecinos si comparten una arista (dos vértices en común, dentro
        de una tolerancia numérica).
        """
        def key(pt: Point) -> Point:
            return (round(pt[0] / self.tol) * self.tol, round(pt[1] / self.tol) * self.tol)

        # edge_key -> lista de triangle ids que tienen esa arista
        edge_map: Dict[Tuple[Point, Point], List[int]] = {}

        for tri in self._triangles:
            for (p, q) in tri.edges():
                pk, qk = key(p), key(q)
                ek = tuple(sorted([pk, qk]))
                edge_map.setdefault(ek, []).append(tri.id)

        graph: Dict[int, List[int]] = {t.id: [] for t in self._triangles}
        for ek, ids in edge_map.items():
            if len(ids) == 2:
                a, b = ids
                graph[a].append(b)
                graph[b].append(a)
            elif len(ids) > 2:
                # No debería ocurrir en una malla planar válida; se ignoran
                # duplicados manteniendo únicamente pares únicos.
                unique_ids = sorted(set(ids))
                for i in range(len(unique_ids)):
                    for j in range(i + 1, len(unique_ids)):
                        graph[unique_ids[i]].append(unique_ids[j])
                        graph[unique_ids[j]].append(unique_ids[i])

        # limpiar duplicados y ordenar
        for tid in graph:
            graph[tid] = sorted(set(graph[tid]))

        # asignar a cada Triangle sus vecinos
        for tri in self._triangles:
            tri.neighbors = graph[tri.id]

        self._graph = graph
        return self._graph

    # ------------------------------------------------------------------ #
    # Interfaz pública
    # ------------------------------------------------------------------ #
    def get_triangles(self) -> List[Triangle]:
        """Retorna la lista de objetos Triangle (orden = id creciente)."""
        return sorted(self._triangles, key=lambda t: t.id)

    def get_graph(self) -> Dict[int, List[int]]:
        """Retorna el grafo de adyacencia: {id: [neighbor_ids]}."""
        return self._graph

    def get_centers(self) -> np.ndarray:
        """Retorna un array (20, 2) con el centro de cada triángulo (ordenado por id)."""
        return np.array([t.center for t in self.get_triangles()])

    def get_pentagon_vertices(self) -> np.ndarray:
        """Retorna los 5 vértices del pentágono regular."""
        return self._pentagon_vertices

    def get_automorphisms(self) -> List[np.ndarray]:
        """
        Calcula el grupo de simetría del pentágono (diedral D5, orden 10:
        5 rotaciones + 5 reflexiones) expresado como permutaciones sobre
        los ids de triángulos (0..19).

        Retorna una lista de 10 arrays de permutación (incluye la identidad).
        perm[i] = j  significa: el triángulo i, tras aplicar la transformación,
        ocupa la posición geométrica del triángulo j.
        """
        centers = self.get_centers()  # (20, 2), índice == id porque get_triangles ordena por id
        O = self.origin

        def match_permutation(transformed: np.ndarray) -> np.ndarray:
            perm = np.zeros(self.N_TRIANGLES, dtype=int)
            for i, pt in enumerate(transformed):
                d = np.linalg.norm(centers - pt, axis=1)
                j = int(np.argmin(d))
                if d[j] > 1e-3 * max(self.radius, 1.0):
                    raise RuntimeError(
                        "No se encontró correspondencia geométrica exacta "
                        "para un automorfismo; revisar construcción de la malla."
                    )
                perm[i] = j
            return perm

        automorphisms = []

        # 5 rotaciones (incluye identidad, k=0)
        for k in range(5):
            theta = 2 * math.pi * k / 5
            c, s = math.cos(theta), math.sin(theta)
            R = np.array([[c, -s], [s, c]])
            transformed = (centers - O) @ R.T + O
            automorphisms.append(match_permutation(transformed))

        # 5 reflexiones (eje que pasa por el centro y el vértice 0, más las
        # rotaciones de ese eje)
        v0 = self._pentagon_vertices[0] - O
        axis_angle = math.atan2(v0[1], v0[0])
        for k in range(5):
            theta = axis_angle + 2 * math.pi * k / 5
            # reflexión respecto a la recta que pasa por O con ángulo theta
            c, s = math.cos(2 * theta), math.sin(2 * theta)
            Rf = np.array([[c, s], [s, -c]])
            transformed = (centers - O) @ Rf.T + O
            automorphisms.append(match_permutation(transformed))

        return automorphisms

    def __repr__(self) -> str:  # pragma: no cover
        return f"PentagonMesh(n_triangles={len(self._triangles)}, radius={self.radius})"
