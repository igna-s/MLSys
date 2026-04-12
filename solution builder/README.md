# Solución Track A (Scheduler Híbrido C++)

Este directorio contiene la implementación de la arquitectura descrita en `solucion.md`.

## Estructura

- `main.cpp`: Punto de entrada, lectura de JSON y escritura del resultado.
- `graph_analyzer.h` / `.cpp`: Funcionalidades estáticas (perfilado de tensores, predecesores/sucesores).
- `scheduler.h` / `.cpp`: El motor principal de la heurística. Se encarga de particionar el grafo, agrupar (fusión de nodos), decidir las dimensiones de Tiling ($w, h, k$), y administrar la retención (`tensors_to_retain`).
- `json_io.h` / `.cpp`: Parser/Writer ultra ligero de JSON para evitar dependencias pesadas si no se dispone de Abseil completo. (O si se dispone, se usa `nlohmann/json.hpp` en su defecto).

## Compilación

```bash
mkdir build && cd build
cmake ..
make
```

## Ejecución

```bash
./mlsys <path_to_input.json> <path_to_output.json>
```
