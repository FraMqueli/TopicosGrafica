import numpy as np
from collections import defaultdict
from dummy import Mesh

epsilon = 1e-6

# Implementación proyectando perspectiva de cada luz (más rápido)
def render_shadow_projection(mesh: Mesh, lights: list[dict] | None) -> np.array:
  if lights is None:
    lights = [{
      "type": "directional",
      "pos": np.array([0.0, 0.0, 2.0]),
      "target": np.array([0.0, 0.0, 0.0]),
      "color": np.array([1.0, 1.0, 1.0]),
      "intensity": 0.4,
    }]

  shadow = np.zeros((256, 256), dtype=np.uint8)

  for light in lights:
    lit_pixels = np.ones((256, 256), dtype=np.uint8)
    for face in mesh.faces:
      v0, v1, v2 = mesh.vertices[face]
      projected = []
      if light["type"] == "directional":
        dir = light["target"] - light["pos"]
        projected = [(project_vertex_to_shadow_map(v0, dir), project_vertex_to_shadow_map(v1, dir), project_vertex_to_shadow_map(v2, dir))] 
        if None in projected:
          projected = []
      else:
        projected = check_clipping(v0, v1, v2, light["pos"])
      
      for p_vertexes in projected:
        cover_pixels_in_light_shadow_map(lit_pixels, p_vertexes[0], p_vertexes[1], p_vertexes[2])

    for i in range(0, 256):
      for j in range(0, 256):
        if lit_pixels[i, j] == 1:
          shadow[i, j] = 1
  return shadow

def check_clipping(v0, v1, v2, origin):
  p0 = project_vertex_to_shadow_map(v0, v0 - origin)
  p1 = project_vertex_to_shadow_map(v1, v1 - origin)
  p2 = project_vertex_to_shadow_map(v2, v2 - origin)

  if (p0 is not None and p1 is not None and p2 is not None):
    return [(p0, p1, p2)]
  
  if (p0 is None and p1 is None and p2 is None):
    return []

  if (p0 is None and p1 is not None and p2 is not None):
    return divide_triangle(v0, v1, v2, origin)

  if (p0 is not None and p1 is None and p2 is not None):
    return divide_triangle(v1, v0, v2, origin)
  
  if (p0 is not None and p1 is not None and p2 is None):
    return divide_triangle(v2, v0, v1, origin)
  
  if (p0 is None and p1 is None and p2 is not None):
    return trim_triangle(v0, v1, v2, origin)

  if (p0 is not None and p1 is None and p2 is None):
    return trim_triangle(v1, v2, v0, origin)
  
  if (p0 is None and p1 is not None and p2 is None):
    return trim_triangle(v0, v2, v1, origin)
  
  #uh oh
  return [(p0, p1, p2)]

def divide_triangle(err, v0, v1, origin):
  herr = err[2] - origin[2]
  h0 = v0[2] - origin[2]
  h1 = v1[2] - origin[2]
  mid_u0 = abs(h0) / (herr + abs(h0)) - epsilon
  mid_u1 = abs(h1) / (herr + abs(h1)) - epsilon
  v2 = mid_u0*err + (1-mid_u0)*v0
  v3 = mid_u1*err + (1-mid_u1)*v1

  p0 = project_vertex_to_shadow_map(v0, v0-origin)
  p1 = project_vertex_to_shadow_map(v1, v1-origin)
  p2 = project_vertex_to_shadow_map(v2, v2-origin)
  p3 = project_vertex_to_shadow_map(v3, v3-origin)

  return [
    (p0, p1, p2),
    (p1, p2, p3)
  ]

def trim_triangle(err1, err2, v0, origin):
  herr1 = err1[2] - origin[2]
  herr2 = err2[2] - origin[2]
  h0 = v0[2] - origin[2]
  mid_u1 = abs(h0) / (herr1 + abs(h0)) - epsilon
  mid_u2 = abs(h0) / (herr2 + abs(h0)) - epsilon
  v1 = mid_u1*err1 + (1-mid_u1)*v0
  v2 = mid_u2*err2 + (1-mid_u2)*v0

  p0 = project_vertex_to_shadow_map(v0, v0-origin)
  p1 = project_vertex_to_shadow_map(v1, v1-origin)
  p2 = project_vertex_to_shadow_map(v2, v2-origin)

  return [
    (p0, p1, p2)
  ]

def project_vertex_to_shadow_map(vertex, direction):
  if vertex[2] > 0 and direction[2] > 0 or vertex[2] < 0 and direction[2] < 0 or vertex[2] != 0 and direction[2] == 0:
    return None
  t = -vertex[2]/direction[2]
  return vertex + t*direction 

def cover_pixels_in_light_shadow_map(lit_pixels, v0, v1, v2):
  if v0 is None or v1 is None or v2 is None:
    return
  pixel_size = 1.2/256
  # Obtenemos los índices de pixel correspondientes a los vértices del triángulo
  p0, p1, p2 = ((v0, v1, v2) + np.array([0.6, 0.6, 0])) / pixel_size - np.array([0.5, 0.5, 0])
  x_min = int(min(p0[0], p1[0], p2[0]))
  x_max = int(max(p0[0], p1[0], p2[0])) + 1
  y_min = int(min(p0[1], p1[1], p2[1]))
  y_max = int(max(p0[1], p1[1], p2[1])) + 1

  for i in range(max(y_min, 0), min(y_max+1, 256)):
    for j in range(max(x_min, 0), min(x_max+1, 256)):
      if(lit_pixels[255 - i, j] == 0):
        continue
      if(point_in_triangle(np.array([j, i, 0]), p0, p1, p2)):
        lit_pixels[255 - i][j] = 0

def point_in_triangle(p, a, b, c):
  # Basado en https://gamedev.stackexchange.com/a/23745
  v0 = b - a
  v1 = c - a
  v2 = p - a
  d00 = np.dot(v0, v0)
  d01 = np.dot(v0, v1)
  d11 = np.dot(v1, v1)
  d20 = np.dot(v2, v0)
  d21 = np.dot(v2, v1)
  denom = d00 * d11 - d01 * d01
  if denom == 0:
    return False
  v = (d11 * d20 - d01 * d21) / denom
  w = (d00 * d21 - d01 * d20) / denom
  u = 1.0 - v - w

  if v < 0 or v > 1 or w < 0 or w > 1 or u < 0 or u > 1:
    return False
  
  return True
  
# Implementación ray tracing (más lento, pero es lo que probé primero :P)
def render_shadow_ray(mesh: Mesh, lights: list[dict] | None) -> np.array:
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
      x_coord = pixel_size*(j + 0.5) - 0.6
      y_coord = pixel_size*(i + 0.5) - 0.6
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
          
  return shadow

def ray_triangle_intersect(ray_origin, ray_direction, v0, v1, v2):
  # Implementación basada en descripción del algoritmo en Wikipedia
  # https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
  edge1 = v1 - v0
  edge2 = v2 - v0

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

  if v < -epsilon or u+v-1 > epsilon:
    return False
  
  return True
