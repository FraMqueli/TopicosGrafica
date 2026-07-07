# Sistema de generación y optimización de marcadores pentagonales

Genera, evalúa y optimiza marcadores binarios definidos sobre un pentágono
subdividido en 20 triángulos, para usarlos como referencias visuales en
experimentos de Structure-from-Motion (COLMAP). El sistema no modifica ni
depende del pipeline interno de COLMAP.

## Instalación

```bash
pip install -r requirements.txt
```

## Uso rápido (pipeline completo)

```bash
python main.py --n 12 --iterations 2000 --out ./output --debug
```

Esto genera 12 marcadores optimizados y diversos, los exporta como
`marker_01.png ... marker_12.png` en `./output`, y además:
- `output/debug/` con vistas de depuración (triangulación, ids, bits activos)
- `output/markers_metadata.json` con los bits, score total y el detalle por
  métrica de cada marcador (trazabilidad del experimento)

Parámetros principales de `main.py`:

| Flag             | Descripción                                                  | Default   |
|------------------|---------------------------------------------------------------|-----------|
| `--n`            | Número final de marcadores a exportar                        | 12        |
| `--pool`         | Tamaño del pool de candidatos antes de filtrar por diversidad| `2*n`     |
| `--iterations`   | Iteraciones de simulated annealing por marcador               | 2000      |
| `--min-hamming`  | Distancia Hamming mínima exigida entre marcadores             | 6         |
| `--img-size`     | Tamaño en píxeles de cada imagen exportada                    | 800       |
| `--seed`         | Semilla aleatoria (reproducibilidad)                          | aleatoria |
| `--debug`        | Exporta también vistas de depuración                          | off       |

## Arquitectura

```
geometry.py   -> estructura geométrica fija (pentágono, 20 triángulos, grafo, simetrías D5)
marker.py     -> representación binaria del marcador (20 bits) y operaciones sobre bits
evaluator.py  -> score(marker, geometry, existing_markers) con 6 métricas de calidad
optimizer.py  -> simulated annealing + generación de población + selección diversa
render.py     -> marcador -> imagen PNG, dataset export, vista de depuración
main.py       -> pipeline de extremo a extremo (CLI)
test_system.py-> pruebas de humo de todo el sistema
```

Cada módulo solo conoce la interfaz pública de los demás, no su
implementación interna, de modo que se pueden reemplazar por separado
(por ejemplo, cambiar la estrategia de optimización sin tocar geometry.py
ni render.py).

## Métricas de calidad (`evaluator.score`)

| Métrica         | Qué mide                                                              |
|-----------------|-------------------------------------------------------------------------|
| `balance`       | Cercanía a una distribución 50/50 de bits 0 y 1                        |
| `transitions`   | Fracción de aristas del grafo con bits distintos (riqueza de textura)  |
| `symmetry`      | Ausencia de invariancia bajo rotaciones/reflexiones del pentágono (D5) |
| `connectivity`  | Ausencia de regiones homogéneas grandes (blobs) vía BFS                |
| `orientation`   | Que el patrón sea distinguible bajo cualquier rotación (pose única)    |
| `separability`  | Distancia Hamming mínima respecto a otros marcadores ya seleccionados  |

El score final es un promedio ponderado (pesos configurables vía
`EvaluatorWeights`), normalizado en `[0, 1]`.

## Uso programático

```python
from geometry import PentagonMesh
from optimizer import generate_population, select_diverse, AnnealingConfig
from render import export_dataset

geo = PentagonMesh()
pool = generate_population(geo, n=20, config=AnnealingConfig(n_iterations=2000), seed=0)
final = select_diverse(pool, k=12)
export_dataset(final, geo, "./output")
```

## Notas geométricas

Cada uno de los 5 "gajos" del pentágono (centro + dos vértices consecutivos)
se subdivide en 4 sub-triángulos mediante los puntos medios de sus lados
(subdivisión clásica 1→4), dando 5×4 = 20 triángulos con adyacencia bien
definida. Los automorfismos del pentágono regular (5 rotaciones + 5
reflexiones, grupo diedral D5) se calculan geométricamente y se usan para
penalizar simetrías y garantizar una orientación distinguible.

## Tests

```bash
python test_system.py
# o, si se dispone de pytest:
python -m pytest test_system.py -v
```
