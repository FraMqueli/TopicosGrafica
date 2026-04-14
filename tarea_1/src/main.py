from dummy import *
from flat_shading import flat_shadding
import utils
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


LIGHTS = [
    {
        "type": "directional",
        "pos": np.array([1.0, 0.5, 2.0]),
        "target": np.array([0.0, 0.0, 0.0]),
        "color": np.array([1.0, 0.0, 0.0]),
        "intensity": 0.4,
    },  # red
    {
        "type": "point",
        "pos": np.array([-1.0, 0.3, 1.0]),
        "color": np.array([0.0, 0.0, 1.0]),
        "intensity": 1.4,
    },  # blue
    {
        "type": "point",
        "pos": np.array([0.0, -2.0, 1.0]),
        "color": np.array([0.0, 1.0, 0.0]),
        "intensity": 1.4,
    },  # green
]


def process_files(npz_files: list[Path], out_dir: Path):
    log.info(
        "Processando %i archivos. Output será guardado en %s", len(
            npz_files), out_dir
    )

    rows = []
    renders_dir = out_dir / "renders"
    renders_dir.mkdir(exist_ok=True, parents=True)

    for path in npz_files:
        with np.load(path) as data:
            verts = np.asarray(data["vertices"], dtype=np.float32)
            faces = np.asarray(data["facets"], dtype=np.int32)

        mesh = Mesh(verts, faces)

        # normalize mesh
        mesh = normalize_mesh(mesh)

        # compute stats
        stats = compute_stats(mesh)
        stats["file"] = path.stem

        # calculate flat shadding
        face_colors = flat_shadding(mesh, LIGHTS)

        # do rendering of shadow
        shadow = render_shadow(mesh, lights=LIGHTS)

        # call render for all views
        model_dir = renders_dir / path.stem
        utils.threejs_render(
            mesh.vertices,
            mesh.faces,
            face_colors=face_colors,
            shadow=shadow,
            shadow_z=0,
            views=["persp", "back", "side_left", "top"],
            name=path.stem,
            out_dir=model_dir,
            save_html=True,
        )

        rows.append(stats)

    df = pd.DataFrame(rows)
    csv_path = out_dir / "metrics.csv"
    df.to_csv(csv_path, index=False)
    log.info("Saved metrics to %s", csv_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Evaluar tarea 1")
    parser.add_argument(
        "--models",
        type=Path,
        default=Path("thingi10k-animals"),
        help="Directorio con archivos de 3D Meshes .npz",
    )
    parser.add_argument(
        "--id",
        type=str,
        default=None,
        help="Procesar un solo modelo por ID (ej. 1215157)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results"),
        help="Directorio de salida para CSV de métricas y renders",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limita el numero de modelos procesados",
    )
    args = parser.parse_args()

    if args.id:
        npz_files = [args.models / f"{args.id}.npz"]
    else:
        npz_files = sorted(args.models.rglob("*.npz"))
        if args.limit:
            npz_files = npz_files[: args.limit]

    process_files(npz_files, args.out)
