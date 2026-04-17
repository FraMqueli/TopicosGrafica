from dummy import Mesh
import numpy as np

# Luz difusa
# Flat Shading -> trabajar con las caras


# Entrada
# Mesh normalizado con V número de vértices y F número de faces
# lista de Dic: [lights[type (Str), por(np.ndarray), target(np.ndarray), color((np.ndarray)), intensity(int)]= value]
# Ambient: escalar

# Salida

# Lista de tuplas (face_i, R_i, G_i, B_i), con R,G,B \in [0,1]


# def flat_shadding(mesh: Mesh, lights: list[dict], ambient: float = 0.3) -> np.ndarray:
#    face_colors = np.zeros((len(mesh.faces), 3), dtype=np.int32)
#    face_colors[:, 0] = 1.0
#    return face_colors

def flat_shadding(mesh: Mesh, lights: list[dict], ambient: float = 0.3) -> np.ndarray:
    verts = mesh.vertices
    faces = mesh.faces

    face_colors = np.zeros((len(faces), 3), dtype=np.float32)

    for i, face in enumerate(faces):
        v0, v1, v2 = verts[face]

        # normal de la cara
        n = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(n)
        if norm == 0:
            continue
        n = n / norm

        # centro de la cara
        center = (v0 + v1 + v2) / 3

        color = np.zeros(3, dtype=np.float32)

        for light in lights:
            if light["type"] == "point":
                color += point_source(light, center, n)

            elif light["type"] == "directional":
                color += directional_source(light, n)

        # componente ambiente
        color += ambient

        # clamp final
        face_colors[i] = np.clip(color, 0.0, 1.0)

    return face_colors


def point_source(light, point, normal):
    l_vec = light["pos"] - point
    r = np.linalg.norm(l_vec)
    l = l_vec / r

    I = light["intensity"]
    E = irradiance_E(I, r, l, normal)

    return E * light["color"]


def directional_source(light, normal):
    l = light["target"] - light["pos"]
    l = l / np.linalg.norm(l)

    E = far_irradiance_E(light["intensity"], l, normal)

    return E * light["color"]



def irradiance_E(I: float, r: float, incidence_v, normal_face):
    incidence_v = np.asarray(incidence_v)
    normal_face = np.asarray(normal_face)

    # normalizar
    incidence_v = incidence_v / np.linalg.norm(incidence_v)
    normal_face = normal_face / np.linalg.norm(normal_face)

    # coseno de incidencia
    cos_theta = np.dot(-incidence_v, normal_face)

    return I / (r**2) * max(0.0, cos_theta)


def far_irradiance_E(H, incidence_v, normal_face):
    incidence_v = np.asarray(incidence_v)
    normal_face = np.asarray(normal_face)

    incidence_v = incidence_v / np.linalg.norm(incidence_v)
    normal_face = normal_face / np.linalg.norm(normal_face)

    # coseno de incidencia
    cos_theta = np.dot(-incidence_v, normal_face)

    return H * max(0.0, cos_theta)


def simple_lambertian_reflection(E, k):
    return E * k


def half_vector(l, v):
    return (np.array(l) + np.array(v))/np.linalg.norm((np.array(l) + np.array(v)))


def BP_lambertian_reflection(R, ks, n, h, p, E):
    return (R/np.pi + ks * np.power(max(0, np.dot(n, h)), p))*E


def l_dist(p, x, r):
    return (p-x)/r
