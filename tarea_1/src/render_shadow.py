import numpy as np
from collections import defaultdict
from dummy import Mesh

epsilon = 1e-6

def render_shadow(mesh: Mesh, lights: list[dict] | None) -> np.array:
  if lights is None:
    lights = [{
      "type": "directional",
      "pos": np.array([0.0, 0.0, 2.0]),
      "target": np.array([0.0, 0.0, 0.0]),
      "color": np.array([1.0, 1.0, 1.0]),
      "intensity": 0.4,
    }]

  pixel_size = 1.2/256

  shadow = np.zeros((256, 256), dtype=np.uint8)
  for i in range(256):
      for j in range(256):
        print(i,j)
        x_coord = pixel_size*(i + 0.5) - 0.6
        y_coord = pixel_size*(j + 0.5) - 0.6
        point = np.array([x_coord, y_coord, 0])

        for light in lights:
          obstructed = False
          ray_direction = light["pos"] - point
          if light["type"] == "directional":
            ray_direction = light["target"] - light["pos"]
          
          for face in mesh.faces:
            v0, v1, v2 = mesh.vertices[face]
            if ray_triangle_intersect(point, ray_direction, v0, v1, v2):
              obstructed = True
              break
          
          if not obstructed:
            shadow[i, j] = 1
            break

def ray_triangle_intersect(ray_origin, ray_direction, v0, v1, v2):
  # Implementación basada en descripción del algoritmo en Wikipedia
  # https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
  edge1 = v1 - v0
  edge2 = v2 - v0
  triangle_normal = np.cross(edge1, edge2)
  ray_cross_e2 = np.cross(ray_direction, edge2)
  det = np.dot(edge1, ray_cross_e2)
  if(abs(det) < epsilon):
    return False
  
  inv_det = 1/det
  s = ray_origin - v0
  u = inv_det*np.dot(s, ray_cross_e2)

  if u < -epsilon or u-1 > epsilon:
    return False
  
  s_cross_e1 = np.cross(s, edge1)
  v = inv_det*np.dot(ray_direction, s_cross_e1)

  if u < -epsilon or u+v-1 > epsilon:
    return False
  
  return True
