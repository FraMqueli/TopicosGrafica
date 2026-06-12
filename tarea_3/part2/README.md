# Parte 2 — Clasificación y visualización con OBBs

## Comentarios generales de la visualización

En el caso de que no se realice correctamente la acción (aparezca la gaussiana, ocultar, trasladar, escalar, etc) basta con hacer click en alguna parte de la pantalla para que se actualice y aparezca el cambio.

## Requisitos

```bash
pip install plyfile numpy
```

## 1. Generar los mappings (clasificación de gaussianas)

Desde la carpeta `part2/`:

```bash
python part2.py --method center-in-box --ply 3dgs.ply --labels labels.json --output mapping_center-in-box.json

python part2.py --method extent-aware --ply 3dgs.ply --labels labels.json --output mapping_extent-aware.json --k 2.0
```

Esto genera los JSON con qué gaussianas pertenecen a cada bounding box.

## 2. Lanzar el viewer


```bash
python -m http.server 8000
```

Luego abrir en el browser:

```
http://localhost:8000/part2/viewer_p2.html?method=center-in-box
http://localhost:8000/part2/viewer_p2.html?method=extent-aware
```

## Parámetros URL del viewer

| Parámetro | Valores | Default |
|-----------|---------|---------|
| `method` | `center-in-box` \| `extent-aware` | `center-in-box` |
| `ply` | ruta al archivo PLY | `3dgs.ply` |
| `labels` | ruta al mapping JSON | `mapping_<method>.json` |

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `part2.py` | Script de clasificación |
| `viewer_p2.html` | Viewer interactivo |
| `labels.json` | Anotaciones de la escena (411 bounding boxes) |
| `3dgs.ply` | Escena en formato estándar 3DGS |
| `mapping_*.json` | Output de la clasificación (generado por part2.py) |
