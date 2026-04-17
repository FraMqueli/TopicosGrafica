import numpy as np
from collections import defaultdict


class Mesh:
    """Triangle mesh container."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        self.vertices: np.ndarray = vertices
        self.faces: np.ndarray = faces


def normalize_mesh(mesh: Mesh) -> Mesh:
    verts = mesh.vertices.copy().astype(np.float64)

    bb_min = verts.min(axis=0)
    bb_max = verts.max(axis=0)

    center_xy = (bb_min + bb_max) / 2
    verts[:, 0] -= center_xy[0]
    verts[:, 1] -= center_xy[1]

    verts[:, 2] -= bb_min[2]

    bb_min2 = verts.min(axis=0)
    bb_max2 = verts.max(axis=0)
    extents = bb_max2 - bb_min2
    scale = extents.max()

    if scale > 0:
        verts /= scale

    return Mesh(verts.astype(np.float32), mesh.faces.copy())


def compute_stats(mesh: Mesh) -> dict:
    verts = mesh.vertices
    faces = mesh.faces

    V = verts.shape[0]
    F = faces.shape[0]

    stats = {}
    stats["num_vertices"] = V
    stats["num_faces"] = F

    edge_to_faces = defaultdict(list)
    for fi, face in enumerate(faces):
        for i in range(3):
            v0, v1 = int(face[i]), int(face[(i + 1) % 3])
            edge = (min(v0, v1), max(v0, v1))
            edge_to_faces[edge].append(fi)

    stats["num_edges"] = len(edge_to_faces)

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    face_areas = 0.5 * np.linalg.norm(cross, axis=1)
    stats["surface_area"] = float(face_areas.sum())

    boundary_edge_list = [e for e, fs in edge_to_faces.items() if len(fs) == 1]
    stats["boundary_edges"] = len(boundary_edge_list)

    boundary_adj = defaultdict(list)
    for v0_e, v1_e in boundary_edge_list:
        boundary_adj[v0_e].append(v1_e)
        boundary_adj[v1_e].append(v0_e)

    visited_bv = set()
    num_loops = 0
    for start in boundary_adj:
        if start not in visited_bv:
            stack = [start]
            while stack:
                v = stack.pop()
                if v not in visited_bv:
                    visited_bv.add(v)
                    for nb in boundary_adj[v]:
                        if nb not in visited_bv:
                            stack.append(nb)
            num_loops += 1
    stats["boundary_loops"] = num_loops

    referenced = set(faces.flatten().tolist())
    stats["floating_vertices"] = V - len(referenced)

    face_adj = defaultdict(set)
    for fs in edge_to_faces.values():
        for i in range(len(fs)):
            for j in range(i + 1, len(fs)):
                face_adj[fs[i]].add(fs[j])
                face_adj[fs[j]].add(fs[i])

    visited_faces = np.zeros(F, dtype=bool)
    num_components = 0
    for start in range(F):
        if not visited_faces[start]:
            stack = [start]
            while stack:
                fi = stack.pop()
                if not visited_faces[fi]:
                    visited_faces[fi] = True
                    for nb in face_adj[fi]:
                        if not visited_faces[nb]:
                            stack.append(nb)
            num_components += 1
    stats["num_components"] = num_components

    directed_edge_to_face = {}
    for fi, face in enumerate(faces):
        for i in range(3):
            a, b = int(face[i]), int(face[(i + 1) % 3])
            directed_edge_to_face[(a, b)] = fi

    is_oriented = True
    for (a, b), fs in edge_to_faces.items():
        if len(fs) == 2:
            has_ab = (a, b) in directed_edge_to_face
            has_ba = (b, a) in directed_edge_to_face
            if not (has_ab and has_ba):
                is_oriented = False
                break
    stats["is_oriented"] = is_oriented

    stats["non_manifold_edges"] = sum(
        1 for fs in edge_to_faces.values() if len(fs) >= 3
    )

    vert_to_faces = defaultdict(list)
    for fi, face in enumerate(faces):
        for v in face:
            vert_to_faces[int(v)].append(fi)

    non_manifold_verts = 0
    for v, v_faces in vert_to_faces.items():
        v_faces_set = set(v_faces)
        local_adj = defaultdict(set)
        for fi in v_faces:
            face = faces[fi]
            for i in range(3):
                if int(face[i]) == v:
                    nb = int(face[(i + 1) % 3])
                    e = (min(v, nb), max(v, nb))
                    for fj in edge_to_faces[e]:
                        if fj != fi and fj in v_faces_set:
                            local_adj[fi].add(fj)
                            local_adj[fj].add(fi)

        visited_local = set()
        local_components = 0
        for fi in v_faces:
            if fi not in visited_local:
                stack = [fi]
                while stack:
                    f = stack.pop()
                    if f not in visited_local:
                        visited_local.add(f)
                        for nb_f in local_adj[f]:
                            if nb_f not in visited_local:
                                stack.append(nb_f)
                local_components += 1

        if local_components > 1:
            non_manifold_verts += 1

    stats["non_manifold_vertices"] = non_manifold_verts

    AREA_THRESHOLD = 1e-10
    degenerate = 0
    for fi, face in enumerate(faces):
        if len(set(face.tolist())) < 3:
            degenerate += 1
        elif face_areas[fi] < AREA_THRESHOLD:
            degenerate += 1
    stats["degenerate_faces"] = degenerate

    return stats


# def flat_shadding(mesh: Mesh, lights: list[dict], ambient: float = 0.3) -> np.ndarray:
#     face_colors = np.zeros((len(mesh.faces), 3), dtype=np.int32)
#     face_colors[:, 0] = 1.0
#     return face_colors


#def render_shadow(mesh: Mesh, lights: list[dict] | None) -> np.array:
#    shadow = np.ones((256, 256), dtype=np.uint8)
#    shadow[10:-10, 10:-10] = 0.2
#    return shadow
