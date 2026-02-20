# Arquitectura y Estrategia — MLSys 2026 Track A

## Resumen

Scheduler que minimiza latencia total ejecutando un DAG de operaciones tensoriales en una jerarquía de memoria (Fast/Slow memory). Implementado en C++20, compilado con Zig C++.

**Resultado:** ~15.4M ciclos totales sobre los 5 benchmarks publicados (23x mejora vs baseline ingenuo).

## Arquitectura del Scheduler

### 1. Programación Dinámica O(N²) con Ventana

El particionamiento de ops en subgrafos se resuelve con DP:
- `dp[i]` = mínima latencia acumulada para ejecutar ops `[0..i)`
- Para cada `i`, consideramos particiones `[j..i)` con `j ∈ [max(0, i-W), i)`
- Ventana `W=10` limita complejidad sin perder buenas fusiones
- Para cada partición candidata, buscamos la mejor granularidad `[w,h,k]`

### 2. Evaluador de Latencia Analítico O(1)

Para cada subgrafo candidato con granularidad `[w,h,k]`:

**Cómputo por paso:**
- MatMul: `base_cost / k_steps` por k-step (costo proporcional a fracción de K)
- Pointwise: `base_cost` por tile espacial (solo en k=0)
- Penalty de padding: `max(w,nw) × max(h,nh) / (w×h)` si sub-nativo

**Memoria por paso (modelo roofline):**
- Tensores de entrada: LHS (h×k), RHS (w×k), Pointwise (w×h)
- Snake traversal: LHS reusado en misma fila, RHS reusado en cambio de fila (snake boundary)
- Tensores estacionarios: si el LHS completo (h×K_dim) cabe en fast memory, cargarlo una vez y reusar en todos los k-steps
- Latencia por step: `max(compute, mem_in + mem_out)`

**Fórmula analítica por fila de tiles:**
```
Fila 0, tile 0: max(comp, lhs + rhs + pw + store)
Fila 0, tiles 1..cols-1: max(comp, rhs + pw + store)  // LHS reusado
Filas r>0, tile 0: max(comp, lhs + pw + store)         // RHS reusado del snake
Filas r>0, tiles 1+: max(comp, rhs + pw + store)
```

### 3. Búsqueda de Granularidad

Para cada subgrafo, probamos combinaciones de:
- `w, h`: desde nativo hasta nativo/8, más dimensiones del tensor de salida
- `k`: desde K_dim completo hasta K_dim/8 (potencias de 2)
- Descartamos candidatos que exceden `fast_memory_capacity`
- Probamos tanto snake por filas como por columnas

### 4. Detección de Tensores Efímeros

Tensor producido Y consumido dentro del mismo subgrafo Y cuyo `last_use` no excede el último op del subgrafo → **efímero** (zero capacity, zero memory cost). Esto es critical para permitir fusiones agresivas.

### 5. Retención Inter-Subgrafo

Tensores que serán usados por subgrafos posteriores pueden quedarse en fast memory:
- Ahorra el costo de eviction (no store) y el costo de reload en el siguiente subgrafo
- Priorización: retener primero los tensores más grandes (mayor ahorro de bandwidth)
- Budget: limitado por `fast_memory_capacity`

### 6. Residencia Estacionaria para Split-K

Per PROBLEM.md Ejemplo 5: en modo split-K, el tensor LHS puede cargarse completo (h×K_dim) una sola vez y reusarse en todos los k-steps. Se activa cuando el working set con LHS completo cabe en fast memory. Reduce drásticamente el memory time en k-steps internos.

## Resultados

| Benchmark | Ops | Score | Subgraphs |
|-----------|-----|-------|-----------|
| B1 | 5 | 127,654 | 1 |
| B5 | 19 | 462,334 | ~8 |
| B9 | 32 | 6,421,177 | 16 |
| B13 | 63 | 7,627,735 | ~20 |
| B17 | 103 | 767,764 | ~16 |
| **TOTAL** | | **15,406,665** | |

## Tecnologías

- C++20, compilado con `zig c++ -O3`
- nlohmann/json para I/O
- Sin dependencias externas (portable)
