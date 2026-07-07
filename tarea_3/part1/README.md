# Parte 1

## part1.py

La implementación agrega parámetros adicionales a los indicados en el enunciado. Estos son:

- `extent_iter`: Número de muestras seleccionadas por Gaussiana al usar el método `extent-aware`. Número natural.
- `extent_threshold`: Porcentaje de muestras que deben ser visibles para una máscara para considerar a la Gaussiana como visible al usar el método `extent-aware`. Real entre 0 y 1.
- `seed`: Valor de inicialización para el muestreo estocástico. Número natural.

Para recrear los experimentos presentados en el informe, se recomienda usar los siguientes comandos:

```bash
python3 part1.py --method center --ply data/kitchen_3dgs.ply --colmap_dir data/kitchen/sparse/0/ --masks_dir data/kitchen_masks/ --output labels_center.json --tau_low 0.3 --tau_high 0.7
```

```bash
python3 part1.py --method extent-aware --ply data/kitchen_3dgs.ply --colmap_dir data/kitchen/sparse/0/ --masks_dir data/kitchen_masks/ --output labels_extent_aware_0.5.json --tau_low 0.3 --tau_high 0.7 --extent_iter 20 --extent_threshold 0.01 --seed 0
```

```bash
python3 part1.py --method extent-aware --ply data/kitchen_3dgs.ply --colmap_dir data/kitchen/sparse/0/ --masks_dir data/kitchen_masks/ --output labels_extent_aware_0.01.json --tau_low 0.3 --tau_high 0.7 --extent_iter 20 --extent_threshold 0.01 --seed 0
```

Para comparar los resultados y obtener la diferencia entre los archivos de labels, se usó el script `compare_labels.py` de la siguiente forma:

```bash
python3 compare_labels.py labels_center.json labels_extent_aware_0.5.json
```

```bash
python3 compare_labels.py labels_center.json labels_extent_aware_0.01.json
```

## viewer_p1.html
El viewer acepta los siguientes parámetros de búsqueda adicionales:

- `labelsURL`: Dirección explícita al archivo que se desea utilizar. Ejemplo: `labels/center_labels.json`.
- `methodURL`: Método usado. Dado un método `[x]`, busca un archivo del nombre `[x]_labels.json` en el mismo directorio. Ejemplo: `center`.
- `tau_low`: Redefine `tau_low` al asignar labels, ignorando los labels del archivo de labels elegido. Solo funciona si también se eligió un `tau_high` y uno de `labelsURL` o `methodURL`. Ejemplo: `0.5`.
- `tau_high`: Redefine `tau_high` al asignar labels, ignorando los labels del archivo de labels elegido. Solo funciona si también se eligió un `tau_low` y uno de `labelsURL` o `methodURL`. Ejemplo: `0.5`.