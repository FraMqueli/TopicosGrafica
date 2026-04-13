# Tarea 1

## Setup

Creen un entorno virtual (e.g. conda or venv) e instalen las dependencias:

**Con conda:**
```bash
conda create -n tarea1 python=3.11
conda activate tarea1
pip install -r requirements.txt
playwright install chromium
```

Para verificar que el setup funciona correctamente, ejecuten el script con el modelo de prueba incluido (`cube.npz`):

```bash
python src/main.py --models dataset --out test --id cube
```

Esto debería generar renders en `results/renders/cube/` sin errores.

---

## Ejecución

El pipeline se ejecuta con:

```bash
python src/main.py --models <directorio_con_npz> --out <directorio_salida>
```

O para ejecutar un solo modelo:
```bash
python src/main.py --models <directorio_con_npz> --out <directorio_salida> --id <model_id>
```
