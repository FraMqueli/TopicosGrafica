import json
import tempfile
from pathlib import Path

import numpy as np
import cv2

TEMPLATE = (Path(__file__).parent / "render_template.html").read_text()

# ---------------------------------------------------------------------------
# Camera presets
# ---------------------------------------------------------------------------

# Mesh is normalized to X,Y ∈ [-0.5, 0.5], Z ∈ [0, 1], center (0, 0, 0.5).
# 80% fill → pad by 1/0.8 = 1.25 on each axis.
_PAD = 1 / 0.8  # 1.25
_XY = 0.5 * _PAD  # 0.625  →  X,Y range [-0.625,  0.625]
_ZLO = 0.5 - 0.5 * _PAD  # -0.125 →  Z range  [-0.125,  1.125]
_ZHI = 0.5 + 0.5 * _PAD  #  1.125

_C = dict(x=0, y=0, z=0.5)  # look-at: mesh center

# Scene ranges applied to orthographic views for consistent zoom
ORTHO_RANGES = dict(
    xaxis=dict(range=[-_XY, _XY]),
    yaxis=dict(range=[-_XY, _XY]),
    zaxis=dict(range=[_ZLO, _ZHI]),
    aspectmode="cube",
)

CAMERAS: dict[str, dict] = {
    "front": dict(
        pose=dict(eye=dict(x=0, y=1, z=0.5), center=_C, up=dict(x=0, y=0, z=1)),
        projection="orthographic",
    ),
    "back": dict(
        pose=dict(eye=dict(x=0, y=-1, z=0.5), center=_C, up=dict(x=0, y=0, z=1)),
        projection="orthographic",
    ),
    "side": dict(
        pose=dict(eye=dict(x=1, y=0, z=0.5), center=_C, up=dict(x=0, y=0, z=1)),
        projection="orthographic",
    ),
    "side_left": dict(
        pose=dict(eye=dict(x=-1, y=0, z=0.5), center=_C, up=dict(x=0, y=0, z=1)),
        projection="orthographic",
    ),
    "top": dict(
        pose=dict(eye=dict(x=0, y=0, z=2), center=_C, up=dict(x=0, y=1, z=0)),
        projection="orthographic",
    ),
    # Perspective: Plotly FOV=45°, 80% fill of bounding sphere (r=√0.75≈0.866)
    # d = r / sin(0.8 × FOV/2) = 0.866 / sin(18°) ≈ 2.80, along (1,-1,1)/√3
    "persp": dict(
        pose=dict(eye=dict(x=1.62, y=-1.62, z=2.12), center=_C, up=dict(x=0, y=0, z=1)),
        projection="perspective",
    ),
}


def _mesh_to_buffers(
    vertices: np.ndarray, faces: np.ndarray, face_colors: np.array
) -> dict:

    verts = vertices.astype(np.float32)
    positions = verts[faces].reshape(-1, 3)

    colors = face_colors.astype(np.float32)
    colors = np.repeat(colors, 3, axis=0)

    return {
        "positions": positions.flatten().tolist(),
        "colors": colors.flatten().tolist(),
    }


def _shadow_to_texture(shadow: np.ndarray) -> dict:
    """
    Convert (H, W) float shadow map to a raw RGBA Uint8Array for Three.js
    DataTexture. Shadow value 1=lit → white, 0=dark → grey.
    """
    h, w = shadow.shape
    # shadow ∈ [0,1]: 0=fully shadowed, 1=lit
    # map to alpha: shadowed areas are semi-transparent dark overlay
    alpha = ((1.0 - shadow) * 180).astype(np.uint8)  # max opacity 180/255

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = 160  # R
    rgba[:, :, 1] = 160  # G
    rgba[:, :, 2] = 160  # B
    rgba[:, :, 3] = alpha

    # Flip vertically — Three.js DataTexture origin is bottom-left
    rgba = rgba[::-1]
    return {"pixels": rgba.flatten().tolist(), "size": w}


def _build_html(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_colors: np.array,
    shadow: None | np.ndarray,
    shadow_z: float,
    width: int,
    height: int,
    camera: dict,
) -> str:

    mesh_data = _mesh_to_buffers(vertices, faces, face_colors)

    shadow_data = "null"
    if shadow is not None:
        shadow_data = json.dumps(_shadow_to_texture(shadow))

    cam_raw = camera["pose"]
    cam = {
        "eye": cam_raw["eye"],
        "center": cam_raw["center"],
        "up": cam_raw["up"],
    }

    html = TEMPLATE
    html = html.replace("__MESH_DATA__", json.dumps(mesh_data))
    html = html.replace('"__CAMERA_TYPE__"', json.dumps(camera["projection"]))
    html = html.replace("__CAM__", json.dumps(cam))
    html = html.replace("__WIDTH__", str(width))
    html = html.replace("__HEIGHT__", str(height))
    html = html.replace("__SHADOW_DATA__", shadow_data)
    html = html.replace("__SHADOW_Z__", str(shadow_z))
    return html


# ---------------------------------------------------------------------------
# Playwright render
# ---------------------------------------------------------------------------


def _screenshot(
    html: str, path: Path, width: int, height: int, use_tempfile: bool
) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": height})

        if use_tempfile:
            with tempfile.NamedTemporaryFile(
                suffix=".html", delete=False, mode="w", encoding="utf-8"
            ) as f:
                f.write(html)
                fout = Path(f.name)
        else:
            fout = path.with_suffix(".html")
            with fout.open("w", encoding="utf-8") as f:
                f.write(html)
                fout = fout.resolve()

        page.goto(fout.as_uri())
        page.wait_for_function("() => window.__RENDER_DONE__ === true", timeout=15000)
        page.screenshot(path=str(path))
        browser.close()

        if use_tempfile:
            fout.unlink()


def threejs_render(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_colors: np.array,
    views: list[str],
    out_dir: Path,
    name: str,
    shadow: None | np.ndarray = None,
    shadow_z: float = 0,
    width: int = 800,
    height: int = 800,
    save_html: bool = False,
) -> list[Path]:
    """
    Render mesh views to PNG using Three.js + Playwright (headless Chromium).

    Produces the same camera presets as render_views() but with exact
    Three.js cameras (OrthographicCamera / PerspectiveCamera).
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for view in views:
        camera = CAMERAS[view]
        html = _build_html(
            vertices, faces, face_colors, shadow, shadow_z, width, height, camera
        )
        path = out_dir / f"{name}_{view}.png"
        _screenshot(html, path, width, height, use_tempfile=not save_html)
        print(f"Saved {path}")
        saved.append(path)

    return saved


def blur_shadow(mask: np.ndarray, blur_radius: float = 10.0) -> np.ndarray:
    """Convert a binary raster mask into a soft shadow map.
    Dilates by blur_radius pixels, then applies a Gaussian blur.
    """
    r = int(blur_radius)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    mask = cv2.dilate(mask.astype(np.uint8), kernel)
    shadow = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=blur_radius)

    return shadow
