"""
main.py
=======

Pipeline de extremo a extremo del sistema:

    geometry.py -> marker.py -> evaluator.py -> optimizer.py -> render.py

Genera un conjunto de N marcadores pentagonales optimizados y diversos,
y los exporta como imágenes PNG listas para usar en experimentos COLMAP.

Uso:
    python main.py --n 12 --out ./output --iterations 2000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from geometry import PentagonMesh
from optimizer import generate_population, select_diverse, AnnealingConfig
from evaluator import score, EvaluatorWeights
from render import export_dataset, debug_view, render_marker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera y optimiza marcadores pentagonales para COLMAP."
    )
    parser.add_argument("--n", type=int, default=12,
                         help="Número final de marcadores a exportar (default: 12)")
    parser.add_argument("--pool", type=int, default=None,
                         help="Tamaño del pool de candidatos antes de seleccionar diversidad "
                              "(default: 2x --n)")
    parser.add_argument("--iterations", type=int, default=2000,
                         help="Iteraciones de simulated annealing por marcador (default: 2000)")
    parser.add_argument("--min-hamming", type=int, default=6,
                         help="Distancia Hamming mínima exigida entre marcadores (default: 6)")
    parser.add_argument("--img-size", type=int, default=800,
                         help="Tamaño en píxeles de cada imagen exportada (default: 800)")
    parser.add_argument("--out", type=str, default="./output",
                         help="Carpeta de salida (default: ./output)")
    parser.add_argument("--seed", type=int, default=None,
                         help="Semilla aleatoria (default: aleatoria)")
    parser.add_argument("--debug", action="store_true",
                         help="También exporta vistas de depuración (triangulación, ids, bits)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pool_size = args.pool or max(args.n * 2, args.n + 4)

    print(f"[1/5] Construyendo geometría (pentágono -> 20 triángulos)...")
    geometry = PentagonMesh()
    print(f"      Triángulos: {len(geometry.get_triangles())}, "
          f"automorfismos: {len(geometry.get_automorphisms())}")

    print(f"[2/5] Generando pool de {pool_size} candidatos optimizados "
          f"({args.iterations} iteraciones c/u)...")
    t0 = time.time()
    config = AnnealingConfig(n_iterations=args.iterations, seed=args.seed)
    weights = EvaluatorWeights()
    pool = generate_population(
        geometry, n=pool_size, config=config, weights=weights,
        enforce_diversity=True, min_hamming=args.min_hamming, seed=args.seed,
    )
    print(f"      Pool generado: {len(pool)} marcadores en {time.time() - t0:.1f}s")

    print(f"[3/5] Seleccionando {args.n} marcadores diversos del pool...")
    final_markers = select_diverse(pool, k=args.n, min_hamming=0)
    print(f"      Seleccionados: {len(final_markers)}")

    print("[4/5] Re-evaluando marcadores finales (score conjunto, con separabilidad real)...")
    summary = []
    for i, m in enumerate(final_markers):
        others = [x for x in final_markers if x is not m]
        s, breakdown = score(m, geometry, existing_markers=others,
                              weights=weights, return_breakdown=True)
        summary.append({
            "index": i + 1,
            "bits": "".join(str(b) for b in m.bits),
            "score": round(s, 4),
            "breakdown": {k: round(v, 4) for k, v in breakdown.items()},
        })
        print(f"      marker_{i+1:02d}: score={s:.4f}  bits={''.join(str(b) for b in m.bits)}")

    print(f"[5/5] Exportando imágenes a {out_dir}...")
    paths = export_dataset(final_markers, geometry, str(out_dir), img_size=args.img_size)
    for p in paths:
        print(f"      -> {p}")

    if args.debug:
        debug_dir = out_dir / "debug"
        debug_dir.mkdir(exist_ok=True)
        for i, m in enumerate(final_markers):
            dbg = debug_view(geometry, marker=m, img_size=args.img_size)
            dbg.save(debug_dir / f"debug_marker_{i+1:02d}.png")
        print(f"      Vistas de depuración exportadas a {debug_dir}")

    # guardar metadatos (bits, scores, breakdown) para trazabilidad del experimento
    meta_path = out_dir / "markers_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nMetadatos guardados en {meta_path}")
    print("Listo.")


if __name__ == "__main__":
    main()
