# Tarea 1 — Borrador de Informe
**IIC3912 - Tópicos Avanzados de Gráfica Computacional**
**Integrante(s):** Sofia Rencoret

---

## 1. Portada
*(completar con todos los integrantes del grupo)*

- Curso: IIC3912 — Tópicos Avanzados de Gráfica Computacional
- Tarea: Tarea 1
- Integrantes: ...

---

## 2. Datos Utilizados

El dataset proviene de **Thingi10K**, una colección de modelos 3D obtenidos de la plataforma Thingiverse.
Se utilizó un subconjunto de la categoría **Toys** (85 modelos + 1 cubo de prueba = 86 archivos `.npz`).

El dataset Thingi10K se caracteriza por:
- Gran variedad de geometría y topología (cerrados, con huecos, no manifold, múltiples componentes)
- Modelos generados por usuarios, por lo que su calidad varía considerablemente
- Escalas y orientaciones arbitrarias en el espacio original (necesitan normalización)

**Modelos usados para validar:**
- `cube.npz`: cubo unitario, usado como caso base. Resultados esperados conocidos:
  - 8 vértices, 12 caras, 18 aristas
  - Euler: V − E + F = 8 − 18 + 12 = 2 ✓ (esfera topológica, cerrada)
  - Área superficial = 6.0 ✓
  - 0 aristas de contorno, 0 componentes adicionales, orientación consistente

**Estadísticas representativas del dataset (muestra):**

| Modelo  | Vértices | Caras  | Área sup. | Componentes | Aristas contorno |
|---------|----------|--------|-----------|-------------|-----------------|
| cube    | 8        | 12     | 6.00      | 1           | 0               |
| 81148   | ...      | ...    | ...       | ...         | ...             |

*(completar con estadísticas reales de al menos 5 modelos)*

---

## 3. Implementación

### 3.1 Normalize Mesh

**Objetivo:** transformar el mesh a un espacio canónico para uniformizar comparaciones.

**Algoritmo:**

1. **Bounding Box (AABB):** Se computan el mínimo y máximo por cada eje:
   ```
   bb_min = min(V)  en cada eje
   bb_max = max(V)  en cada eje
   ```

2. **Centrado en X, Y:** Se traslada el centro del bounding box al origen:
   ```
   center_xy = (bb_min + bb_max) / 2
   V_x ← V_x - center_xy[x]
   V_y ← V_y - center_xy[y]
   ```

3. **Apoyo en Z = 0:** Se traslada el modelo para que el vértice más bajo quede en Z = 0:
   ```
   V_z ← V_z - bb_min[z]
   ```

4. **Escalado uniforme:** El lado más largo del bounding box debe medir 1.0. Se escala por el inverso del máximo extent:
   ```
   scale = max(bb_max - bb_min)  (recomputado post-traslación)
   V ← V / scale
   ```

**Resultado:** El bounding box queda en [-0.5, 0.5] × [-0.5, 0.5] × [0, 1], con el lado más largo igual a 1.0. El aspect ratio del modelo se conserva.

---

### 3.2 Compute Stats

Se calculan 12 estadísticas geométricas y topológicas usando únicamente NumPy y la librería estándar de Python.

#### Aristas únicas (`num_edges`)
Se construye un diccionario `edge → [lista de caras]`. Cada arista se representa como tupla ordenada `(min(v0,v1), max(v0,v1))` para que sea no dirigida. El número de llaves del diccionario es el número de aristas únicas.

#### Área superficial (`surface_area`)
Área de un triángulo via producto cruzado (fórmula vista en clase 02):
```
e1 = V1 - V0
e2 = V2 - V0
n = e1 × e2
area = 0.5 * ||n||
```
Se vectoriza sobre todas las caras con NumPy.

#### Aristas de contorno (`boundary_edges`)
Usando el diccionario `edge → [caras]`, las aristas de contorno son aquellas donde `len(caras) == 1` (pertenecen a exactamente una cara). En una malla cerrada manifold, toda arista tiene exactamente 2 caras.

#### Boundary loops (`boundary_loops`)
Las aristas de contorno forman un grafo. Se cuenta el número de **componentes conexas** de ese grafo con BFS, cada componente es un loop de contorno cerrado.

#### Vértices flotantes (`floating_vertices`)
Se construye el conjunto de todos los vértices referenciados por alguna cara. Los flotantes son los que no aparecen: `V - len(referenced)`.

#### Componentes conexas (`num_components`)
BFS sobre el grafo de adyacencia de caras. Dos caras son adyacentes si comparten una arista (se extrae del diccionario `edge → [caras]`).

#### Orientación consistente (`is_oriented`)
Una malla está orientada consistentemente si para cada arista compartida por dos caras, una cara la recorre en dirección `a→b` y la otra en dirección `b→a`. Se construye un diccionario de aristas dirigidas `(a,b) → cara` y se verifica que para cada arista `(a,b)` con 2 caras, tanto `(a,b)` como `(b,a)` existan en el diccionario.

#### Aristas no manifold (`non_manifold_edges`)
Aristas del diccionario con `len(caras) >= 3`.

#### Vértices no manifold (`non_manifold_vertices`)
Para cada vértice `v`, se reúnen las caras que lo contienen y se construye su grafo de adyacencia local (dos caras del fan son adyacentes si comparten una arista que pasa por `v`). Si el número de componentes conexas del fan es > 1, el vértice es no manifold (caso "bowtie" o similares).

#### Caras degeneradas (`degenerate_faces`)
Una cara es degenerada si tiene vértices repetidos (`len(set(face)) < 3`) o si su área es menor a un umbral (`1e-10`).

---

## 4. Verificación y Debugging

### Cubo unitario
El cubo es el caso de verificación principal por tener resultados teóricos conocidos:

| Estadística        | Esperado       | Obtenido | Correcto |
|--------------------|---------------|----------|---------|
| num_vertices       | 8             | 8        | ✓       |
| num_faces          | 12            | 12       | ✓       |
| num_edges          | 18            | 18       | ✓       |
| surface_area       | 6.0           | 6.0      | ✓       |
| boundary_edges     | 0             | 0        | ✓       |
| boundary_loops     | 0             | 0        | ✓       |
| floating_vertices  | 0             | 0        | ✓       |
| num_components     | 1             | 1        | ✓       |
| is_oriented        | True          | True     | ✓       |
| non_manifold_edges | 0             | 0        | ✓       |
| non_manifold_verts | 0             | 0        | ✓       |
| degenerate_faces   | 0             | 0        | ✓       |

**Fórmula de Euler:** V − E + F = 8 − 18 + 12 = **2** (correcto para un sólido topológicamente equivalente a una esfera).

### Normalización
Se verificó con el modelo `81148` que:
- El mayor extent es exactamente 1.0
- El centro XY es (0, 0) con precisión de float32
- El mínimo Z es 0.0

### Uso de IA
Se utilizó Claude Code (claude-sonnet-4-6) como asistente. Se generó el código de `normalize_mesh` y `compute_stats`. La validación se realizó manualmente comparando con los valores teóricos del cubo y verificando la fórmula de Euler. Se verificó que las fórmulas utilizadas coincidan con las vistas en clase (bounding box de clase 03, normales de clase 02).

---

## 5. Contribución por integrante

*(completar)*

---

## 6. Uso de IA

- **Herramienta:** Claude Code, modelo `claude-sonnet-4-6`
- **Uso principal:** implementación de `normalize_mesh` y `compute_stats`, y borrador del informe
- **Validación:** se verificó cada función contra el cubo (valores teóricos conocidos) y contra la fórmula de Euler. Se revisó que la lógica de cada cálculo coincida con las clases del curso (clase 02 para normales/área, clase 03 para bounding box).
- **Modificaciones:** se ajustó el tipo de dato de los vértices a `float64` durante los cálculos para evitar pérdida de precisión, y se convirtió de vuelta a `float32` al retornar.
