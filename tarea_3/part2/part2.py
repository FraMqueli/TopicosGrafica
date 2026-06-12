import argparse
import json
import numpy as np
from plyfile import PlyData


def load_ply(path):
    ply = PlyData.read(path)
    v = ply["vertex"]

    if "x" in v.data.dtype.names:
        centers = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
        scales  = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(np.float64)
        quats   = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float64)
    else:
        centers, scales, quats = _decode_compressed(ply)

    return centers, scales, quats


def _decode_compressed(ply):
    v = ply["vertex"]
    ch = ply["chunk"]
    N = len(v.data)
    CHUNK_SIZE = 256
    ci = np.arange(N) // CHUNK_SIZE         

    pp = v.data["packed_position"].astype(np.uint32)
    ps = v.data["packed_scale"].astype(np.uint32)
    pr = v.data["packed_rotation"].astype(np.uint32)

    def lerp(mn, mx, t): return mn + t * (mx - mn)
    px = ((pp >> 21) & 0x7FF) / 2047.0
    py = ((pp >> 10) & 0x7FF) / 2047.0
    pz = ((pp >>  0) & 0x3FF) / 1023.0
    x = lerp(ch.data["min_x"][ci], ch.data["max_x"][ci], px)
    y = lerp(ch.data["min_y"][ci], ch.data["max_y"][ci], py)
    z = lerp(ch.data["min_z"][ci], ch.data["max_z"][ci], pz)
    centers = np.stack([x, y, z], axis=1)

    sx = ((ps >> 21) & 0x7FF) / 2047.0
    sy = ((ps >> 10) & 0x7FF) / 2047.0
    sz = ((ps >>  0) & 0x3FF) / 1023.0
    scale_x = lerp(ch.data["min_scale_x"][ci], ch.data["max_scale_x"][ci], sx)
    scale_y = lerp(ch.data["min_scale_y"][ci], ch.data["max_scale_y"][ci], sy)
    scale_z = lerp(ch.data["min_scale_z"][ci], ch.data["max_scale_z"][ci], sz)
    scales = np.stack([scale_x, scale_y, scale_z], axis=1)

    SQRT2 = np.sqrt(2.0)
    li = (pr >> 30) & 3                        
    a  = ((pr >> 20) & 0x3FF) / 1023.0 * SQRT2 - SQRT2 / 2
    b  = ((pr >> 10) & 0x3FF) / 1023.0 * SQRT2 - SQRT2 / 2
    c  = ((pr >>  0) & 0x3FF) / 1023.0 * SQRT2 - SQRT2 / 2
    d  = np.sqrt(np.maximum(0.0, 1.0 - a*a - b*b - c*c))


    quats = np.zeros((N, 4))                    
    for li_val, col_order in [(0, [3,0,1,2]),   
                               (1, [2,3,0,1]),    
                               (2, [1,2,3,0]),    
                               (3, [0,1,2,3])]:  
        m = (li == li_val)
        if not m.any():
            continue
        comp = np.column_stack([a[m], b[m], c[m], d[m]])
        quats[m] = comp[:, col_order]

    return centers, scales, quats


def parse_corners(obj):
    for key in ("bounding_box", "corners", "vertices", "bbox_corners", "points", "psr"):
        val = obj.get(key)
        if val is None:
            continue
       
        if isinstance(val, list) and len(val) == 8 and isinstance(val[0], dict):
            return np.array([[p["x"], p["y"], p["z"]] for p in val], dtype=np.float64)
       
        arr = np.array(val, dtype=np.float64)
        if arr.size == 24:
            return arr.reshape(8, 3)
  
    for key in ("bbox", "obb", "box"):
        nested = obj.get(key)
        if isinstance(nested, dict):
            try:
                return parse_corners(nested)
            except ValueError:
                pass
    raise ValueError(f"No se encontraron 8 esquinas en: {list(obj.keys())}")


def obb_from_corners(corners):
    center = corners.mean(axis=0)
    centered = corners - center
    cov = (centered.T @ centered) / len(corners)
    eigvals, eigvecs = np.linalg.eigh(cov)     
    axes = eigvecs.T.copy()                   
    half_sizes = np.sqrt(np.maximum(eigvals, 0))

    if np.linalg.det(eigvecs) < 0:
        axes[0] = -axes[0]
    return center, axes, half_sizes


def build_covariances(scales, quats):

    s = np.exp(scales)                       
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    R = np.zeros((len(scales), 3, 3))
    R[:, 0, 0] = 1 - 2*(y*y + z*z);  R[:, 0, 1] = 2*(x*y - w*z);  R[:, 0, 2] = 2*(x*z + w*y)
    R[:, 1, 0] = 2*(x*y + w*z);      R[:, 1, 1] = 1 - 2*(x*x + z*z); R[:, 1, 2] = 2*(y*z - w*x)
    R[:, 2, 0] = 2*(x*z - w*y);      R[:, 2, 1] = 2*(y*z + w*x);  R[:, 2, 2] = 1 - 2*(x*x + y*y)
    return np.einsum("nj,nij,nkj->nik", s**2, R, R)   # (N,3,3)


def classify_center_in_box(centers, obb_center, axes, half_sizes):
    proj = (centers - obb_center) @ axes.T      
    return np.all(np.abs(proj) <= half_sizes, axis=1)


def classify_extent_aware(centers, Sigma, obb_center, axes, half_sizes, k):
    proj = (centers - obb_center) @ axes.T       
    mask = np.ones(len(centers), dtype=bool)
    for i in range(3):

        var_i = np.einsum("j,njk,k->n", axes[i], Sigma, axes[i])  
        sigma_i = np.sqrt(np.maximum(var_i, 0))
        mask &= np.abs(proj[:, i]) <= half_sizes[i] + k * sigma_i
    return mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["center-in-box", "extent-aware"], required=True)
    p.add_argument("--ply", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--k", type=float, default=2.0,
                   help="Multiplicador sigma para extent-aware (default: 2.0)")
    args = p.parse_args()

    print(f"Cargando {args.ply}...")
    centers, scales, quats = load_ply(args.ply)
    N = len(centers)
    print(f"  {N} gaussianas cargadas")

    with open(args.labels) as f:
        raw = json.load(f)
    objs = raw if isinstance(raw, list) else raw.get("objects", raw.get("labels", []))

    Sigma = None
    if args.method == "extent-aware":
        print("Construyendo matrices de covarianza...")
        Sigma = build_covariances(scales, quats)

    boxes = []
    box_to_gaussians = []

    for obj in objs:
        try:
            corners = parse_corners(obj)
        except ValueError as e:
            print(f"  Advertencia: saltando caja — {e}")
            continue

        center, axes, half_sizes = obb_from_corners(corners)

        if args.method == "center-in-box":
            mask = classify_center_in_box(centers, center, axes, half_sizes)
        else:
            mask = classify_extent_aware(centers, Sigma, center, axes, half_sizes, args.k)

        label = (obj.get("label") or obj.get("category") or obj.get("class")
                 or obj.get("instance_id") or str(len(boxes)))
        boxes.append({
            "label": str(label),
            "center": center.tolist(),
            "axes": axes.tolist(),
            "half_sizes": half_sizes.tolist(),
        })
        box_to_gaussians.append(np.where(mask)[0].tolist())

    result = {
        "method": args.method,
        "parameters": {"k": args.k} if args.method == "extent-aware" else {},
        "n_gaussians": N,
        "n_boxes": len(boxes),
        "boxes": boxes,
        "box_to_gaussians": box_to_gaussians,
    }

    with open(args.output, "w") as f:
        json.dump(result, f)

    counts = [len(g) for g in box_to_gaussians]
    print(f"\n{len(boxes)} cajas procesadas, {N} gaussianas")
    if counts:
        print(f"Gaussianas por caja: min={min(counts)}, max={max(counts)}, "
              f"media={sum(counts)//len(counts)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
