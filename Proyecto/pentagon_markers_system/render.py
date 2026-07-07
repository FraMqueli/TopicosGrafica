"""
render.py
=========

Convierte la representación abstracta (bits + geometría) en imágenes
utilizables en experimentos reales (por ejemplo, para imprimir marcadores
y usarlos como referencias visuales en capturas para COLMAP).

Este módulo NO conoce evaluación ni optimización: solo dibuja.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from geometry import PentagonMesh
from marker import Marker

# Colores por defecto (contraste binario alto)
COLOR_ZERO = (0, 0, 0)        # negro
COLOR_ONE = (255, 255, 255)   # blanco
COLOR_OUTLINE = (128, 128, 128)
COLOR_BG = (255, 0, 0)        # fondo fuera del pentágono (para detectar recorte fácil)


def _world_to_pixel(points: np.ndarray, geometry: PentagonMesh,
                     img_size: int, margin: float = 0.08) -> np.ndarray:
    """
    Transforma coordenadas del mundo (centradas en geometry.origin, radio
    geometry.radius) a coordenadas de píxel dentro de una imagen cuadrada
    de tamaño img_size, dejando un margen relativo.
    """
    R = geometry.radius
    usable = img_size * (1 - 2 * margin)
    scale = usable / (2 * R)
    cx = cy = img_size / 2.0

    px = cx + (points[:, 0] - geometry.origin[0]) * scale
    # invertir eje Y: en imágenes, Y crece hacia abajo
    py = cy - (points[:, 1] - geometry.origin[1]) * scale
    return np.stack([px, py], axis=1)


def render_marker(marker: Marker,
                   geometry: PentagonMesh,
                   img_size: int = 800,
                   color_zero: Tuple[int, int, int] = COLOR_ZERO,
                   color_one: Tuple[int, int, int] = COLOR_ONE,
                   draw_outline: bool = True,
                   outline_color: Tuple[int, int, int] = COLOR_OUTLINE,
                   background: Optional[Tuple[int, int, int]] = (200, 200, 200),
                   label: Optional[str] = None) -> Image.Image:
    """
    Dibuja el marcador: cada triángulo se colorea según su bit
    (0 -> color_zero, 1 -> color_one), se mantiene el contorno del pentágono.

    Parameters
    ----------
    marker : Marker
    geometry : PentagonMesh
    img_size : int
        Tamaño (en píxeles) del lado de la imagen cuadrada de salida.
    color_zero, color_one : tuple(int,int,int)
        Colores RGB para bit=0 y bit=1.
    draw_outline : bool
        Si se dibuja el contorno de cada triángulo.
    outline_color : tuple(int,int,int)
    background : tuple(int,int,int) o None
        Color de fondo fuera del pentágono. None = transparente.
    label : str, opcional
        Texto a dibujar debajo del marcador (por ejemplo, un ID).

    Returns
    -------
    PIL.Image.Image
    """
    mode = "RGBA" if background is None else "RGB"
    bg = (0, 0, 0, 0) if background is None else background
    img = Image.new(mode, (img_size, img_size), bg)
    draw = ImageDraw.Draw(img)

    triangles = geometry.get_triangles()
    bits = marker.bits

    for tri in triangles:
        pixel_verts = _world_to_pixel(tri.vertices, geometry, img_size)
        polygon = [tuple(p) for p in pixel_verts]
        fill = color_one if bits[tri.id] else color_zero
        draw.polygon(polygon, fill=fill,
                     outline=outline_color if draw_outline else None,
                     width=1 if draw_outline else 0)

    # contorno general del pentágono, para reforzar el borde exterior
    pent_verts_px = _world_to_pixel(geometry.get_pentagon_vertices(), geometry, img_size)
    draw.polygon([tuple(p) for p in pent_verts_px], outline=(0, 0, 0), width=3)

    if label:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        text_pos = (img_size * 0.02, img_size * 0.94)
        draw.text(text_pos, label, fill=(0, 0, 0), font=font)

    return img


def export_dataset(markers: Sequence[Marker],
                    geometry: PentagonMesh,
                    output_dir: str,
                    prefix: str = "marker",
                    img_size: int = 800,
                    **render_kwargs) -> list:
    """
    Exporta múltiples marcadores como imágenes PNG numeradas:
        marker_01.png, marker_02.png, ..., marker_NN.png

    Parameters
    ----------
    markers : list[Marker]
    geometry : PentagonMesh
    output_dir : str
        Carpeta de destino (se crea si no existe).
    prefix : str
        Prefijo del nombre de archivo.
    img_size : int
    render_kwargs :
        Argumentos adicionales pasados a render_marker (colores, etc.).

    Returns
    -------
    list[str]
        Rutas de los archivos generados.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    paths = []
    width = len(str(len(markers)))
    for i, marker in enumerate(markers, start=1):
        filename = f"{prefix}_{str(i).zfill(max(2, width))}.png"
        filepath = out_path / filename
        img = render_marker(marker, geometry, img_size=img_size,
                             label=f"{prefix}_{str(i).zfill(max(2, width))}",
                             **render_kwargs)
        img.save(filepath)
        paths.append(str(filepath))

    return paths


def debug_view(geometry: PentagonMesh,
                marker: Optional[Marker] = None,
                img_size: int = 800,
                show_ids: bool = True,
                show_adjacency: bool = True) -> Image.Image:
    """
    Visualización de depuración: triangulación, conectividad del grafo y,
    opcionalmente, los bits activos de un marcador.

    Parameters
    ----------
    geometry : PentagonMesh
    marker : Marker, opcional
        Si se entrega, colorea los triángulos según sus bits (semitransparente
        para no ocultar los ids/lineas de adyacencia).
    img_size : int
    show_ids : bool
        Dibuja el id de cada triángulo en su centro.
    show_adjacency : bool
        Dibuja líneas entre los centros de triángulos vecinos.

    Returns
    -------
    PIL.Image.Image
    """
    img = Image.new("RGB", (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    triangles = geometry.get_triangles()
    graph = geometry.get_graph()

    # triángulos (y bits si hay marcador)
    for tri in triangles:
        pixel_verts = _world_to_pixel(tri.vertices, geometry, img_size)
        polygon = [tuple(p) for p in pixel_verts]
        if marker is not None:
            fill = (60, 60, 60) if marker.bits[tri.id] else (230, 230, 230)
        else:
            fill = (255, 255, 255)
        draw.polygon(polygon, fill=fill, outline=(0, 0, 0), width=1)

    # líneas de adyacencia entre centros
    if show_adjacency:
        centers_px = _world_to_pixel(geometry.get_centers(), geometry, img_size)
        drawn = set()
        for tid, neighbors in graph.items():
            for nid in neighbors:
                key = tuple(sorted((tid, nid)))
                if key in drawn:
                    continue
                drawn.add(key)
                p1 = tuple(centers_px[tid])
                p2 = tuple(centers_px[nid])
                draw.line([p1, p2], fill=(200, 0, 0), width=2)

    # ids de triángulos
    if show_ids:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        centers_px = _world_to_pixel(geometry.get_centers(), geometry, img_size)
        for tri in triangles:
            cx, cy = centers_px[tri.id]
            text_color = (255, 255, 0) if marker is not None and marker.bits[tri.id] else (0, 0, 255)
            draw.text((cx - 5, cy - 5), str(tri.id), fill=text_color, font=font)

    return img
