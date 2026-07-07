"""
marker.py
=========

Define cómo se representa un marcador sobre la estructura geométrica fija
(PentagonMesh de geometry.py).

Un marcador es, exclusivamente, un vector binario de tamaño 20 (un bit por
triángulo) más un score opcional (calculado externamente por evaluator.py).

Este módulo NO conoce geometría, evaluación ni optimización: solo sabe
manipular el estado binario.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

N_BITS = 20


@dataclass
class Marker:
    """Representación del estado de un marcador binario de 20 bits."""

    bits: np.ndarray
    score: Optional[float] = None

    def __post_init__(self) -> None:
        self.bits = np.asarray(self.bits, dtype=np.uint8).reshape(-1)
        if self.bits.shape[0] != N_BITS:
            raise ValueError(f"bits debe tener tamaño {N_BITS}, recibido {self.bits.shape[0]}")
        if not np.all(np.isin(self.bits, [0, 1])):
            raise ValueError("bits debe contener únicamente valores 0 o 1")

    # ------------------------------------------------------------------ #
    # Constructores
    # ------------------------------------------------------------------ #
    @staticmethod
    def random(rng: Optional[np.random.Generator] = None,
               p_one: float = 0.5) -> "Marker":
        """
        Genera un marcador aleatorio válido.

        Parameters
        ----------
        rng : np.random.Generator, opcional
            Generador de números aleatorios. Si no se entrega, se crea uno nuevo.
        p_one : float
            Probabilidad de que un bit individual sea 1 (por defecto 0.5).
        """
        rng = rng or np.random.default_rng()
        bits = (rng.random(N_BITS) < p_one).astype(np.uint8)
        return Marker(bits=bits)

    @staticmethod
    def from_int(value: int) -> "Marker":
        """Construye un marcador a partir de un entero interpretado en binario (20 bits)."""
        if not (0 <= value < (1 << N_BITS)):
            raise ValueError(f"value debe estar en [0, {1 << N_BITS})")
        bits = np.array([(value >> i) & 1 for i in range(N_BITS)], dtype=np.uint8)
        return Marker(bits=bits)

    # ------------------------------------------------------------------ #
    # Operaciones
    # ------------------------------------------------------------------ #
    def flip(self, i: int) -> "Marker":
        """Invierte el bit en la posición i (in-place) y retorna self."""
        if not (0 <= i < N_BITS):
            raise IndexError(f"índice fuera de rango: {i}")
        self.bits[i] = 1 - self.bits[i]
        self.score = None  # el score queda invalidado tras modificar el estado
        return self

    def flip_many(self, indices) -> "Marker":
        """Invierte varios bits a la vez (in-place) y retorna self."""
        for i in indices:
            self.flip(i)
        return self

    def copy(self) -> "Marker":
        """Retorna una copia independiente del marcador."""
        return Marker(bits=self.bits.copy(), score=self.score)

    def hamming(self, other: "Marker") -> int:
        """Distancia de Hamming respecto a otro marcador."""
        if not isinstance(other, Marker):
            raise TypeError("other debe ser una instancia de Marker")
        return int(np.sum(self.bits != other.bits))

    def validate(self, min_ones: int = 4, max_ones: int = 16) -> bool:
        """
        Aplica restricciones básicas de validez:
            - balance: número de unos dentro de un rango razonable
              (evita marcadores triviales, todo-ceros o todo-unos).

        Retorna True si el marcador cumple las restricciones.
        """
        n_ones = int(self.bits.sum())
        return min_ones <= n_ones <= max_ones

    def as_tuple(self) -> tuple:
        """Representación inmutable y hasheable del marcador (útil para sets/diccionarios)."""
        return tuple(int(b) for b in self.bits)

    def to_int(self) -> int:
        """Convierte el marcador a un entero (para almacenamiento/IDs compactos)."""
        value = 0
        for i, b in enumerate(self.bits):
            value |= int(b) << i
        return value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Marker):
            return NotImplemented
        return bool(np.array_equal(self.bits, other.bits))

    def __hash__(self) -> int:
        return hash(self.as_tuple())

    def __repr__(self) -> str:  # pragma: no cover
        bitstring = "".join(str(b) for b in self.bits)
        score_str = f"{self.score:.4f}" if self.score is not None else "None"
        return f"Marker(bits={bitstring}, score={score_str})"
