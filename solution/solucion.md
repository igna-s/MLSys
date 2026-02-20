# Arquitectura y Estrategia de Solución para MLSys 2026 - Track A (Sistemas)

Este documento presenta una propuesta detallada de arquitectura, algoritmos y estrategias para abordar el Track A del **MLSys 2026 Scheduling Challenge**, donde el objetivo es generar un "schedule" (plan de ejecución de un grafo computacional) que minimice la latencia total del programa, respetando la estricta capacidad de la Memoria Rápida (Fast Memory).

## 1. Análisis del Problema

El problema central es el *Memory-Constrained Operator Scheduling and Fusion*. La ejecución de un Grafo Acíclico Dirigido (DAG) de operaciones tensoriales debe mapearse a una jerarquía de memoria (Fast/Slow memory) bajo un modelo estricto de **working set**.

Las decisiones clave que debe tomar nuestro programa (el *scheduler*) para cada subgrafo (grupo de operaciones fusionadas) son:

1.  **Agrupación (Grouping / Fusion):** Qué nodos del DAG ejecutar juntos (como un único paso/subgrafo). Esto convierte el flujo de datos intermedio en memoria "efímera", ahorrando capacidad y tiempo de lectura/escritura en la memoria rápida.
2.  **Granularidad de Ejecución (Execution Granularity - $w, h, k$):** A qué tamaño de bloque dividir (tiling) la ejecución del subgrafo. Un tile más pequeño reduce la huella en memoria rápida (el working set temporal), pero si es menor a la "granularidad nativa" del hardware, el cómputo incurre en ineficiencias de *padding* y múltiples pasadas.
3.  **Gestión de Residencia (Tensors to Retain):** Qué tensores mantener en la memoria rápida entre la ejecución de un subgrafo y el siguiente, para evitar el costo de enviarlos a la memoria lenta (Slow Memory) y volver a cargarlos mas adelante.
4.  **Orden de Recorrido Espacial (Traversal Order):** Cómo ordenar los bloques de ejecución (spatial tiles) dentro de un subgrafo específico (ej. "Snake order", "Zig-Zag"). Esto permite la reutilización de datos intra-subgrafo (ej. reusar una tira de fila o columna sin tener que recargarla repetidas veces cuando la granularidad es pequeña).

El objetivo es minimizar la métrica fundamental de latencia basada en el modelo "Roofline": La latencia de cada subgrafo es el **máximo** entre su tiempo de Cómputo (Compute Time) y su tiempo de Memoria (Memory Transfer Time = Total Bytes Moved / Slow Memory Bandwidth). El objetivo global es minimizar la suma estricta de las latencias de todos los subgrafos.

## 2. Arquitectura del Scheduler (`mlsys` binario)

Dado que es un problema NP-Hard (involucra particionamiento de grafos, allocation de memoria, dimensionamiento de tiles de ejecución en un espacio 3D muy grande), una solución basada en reglas estáticas será subóptima. Se propone una arquitectura dividida en tres etapas o motores (engines):

### Etapa 1: Preprocesamiento y Análisis Estático (Graph Analyzer)

El primer paso es parsear el JSON de entrada construir estructuras de datos enriquecidas.
*   **Análisis de Camino Crítico (Critical Path):** Calcular los niveles topológicos y las dependencias temporales "más largas".
*   **Perfilado de Tensores (Tensor Profiling):** Identificar qué tensores son "pesados" (aquellos que, si se llevan a memoria lenta, dominan el tiempo de ejecución) frente a tensores pequeños. Determinar tiempo de vida o lapso lógico de uso de cada tensor (liveness).
*   **Detección de Patrones (Pattern Matching):** Detectar sub-estructuras estándar que son altamente eficientes al ser agrupadas:
    *   Cadenas consecutivas de operaciones *Pointwise* (ej. activaciones, normalizaciones).
    *   Patrones de *Split-K* / *FlashAttention*: Multiplicaciones de matrices en cadena (como el Ej. 5 del spec) donde iterar sobre la dimensión interna $K$ de acortar la huella temporal en memoria.
    *   Patrones "Diamante": Donde un tensor intermedio (skip connection) se divide y converge más adelante. Esto es vital para decidir si conviene re-computar y descartar, o almacenar.

### Etapa 2: Motor de Particionamiento (Graph Partitioning Engine)

Esta etapa determina la decisión fundamental: la lista de subgrafos. La arquitectura usará una técnica híbrida: **Búsqueda Golotosa Heurística combinada con Beam Search o Simulated Annealing** para evadir óptimos locales.

*   **Algoritmo Principal - Programación Dinámica (Dynamic Programming) sobre cortes:** Las combinaciones de fusión se modelan como encontrar cortes óptimos en el DAG. Se exploran las particiones viables (donde los tensores inter-subgrafo no generan dependencias cíclicas si se fusionan ciertos nodos).
*   **Heurística de Fusión (Fusion Heuristics):** Intentaremos fusionar nodos de manera agresiva siempre y cuando la suma de la capacidad de inputs/outputs del subgrafo *al usar un tile de tamaño nativo* no exceda la capacidad.
*   **Gestión del "OOM" (Out of Memory):** El mayor castigo de agrupar nodos masivamente es que el tamaño del Working Set rompe la barrera de `fast_memory_capacity`. El motor de particionamiento generará agrupaciones "muy grandes", confiando en que la **Etapa 3** aplicará *tiling* fino o *split-k* para que quepan. Si la Etapa 3 fracasa, el particionamiento debe dividir el subgrafo en subgrafos más pequeños (hacer "fallback" gradual del tamaño del mega-grupo hacia la estrategia de spill out base).

### Etapa 3: Motor de Tiling (Micro-Scheduler & Tiling Solver)

Para cada candidato de subgrafo propuesto por la Etapa 2, este motor intenta encontrar los valores $[w, h, k]$ óptimos y decidir los `tensors_to_retain` y `traversal_order`.

*   Esta es una búsqueda en el espacio de parámetros $(W, H, K)$ donde las variables están constreñidas por los tamaños originales de los tensores base.
*   Dado el límite estricto de tiempo del binario (ej. timeouts entre 2 a 120 segundos), resolver las ecuaciones No-Lineales en Enteros (MINLP) puede ser muy lento en C++.
*   **Solución Algorítmica (Grid Search Optimizada):**
    1.  Empezar probando las dimensiones nativas de `native_granularity`. Si la memoria requerida cumple `working_set <= fast_memory_capacity`, aceptalo (es el óptimo global de cómputo para ese subgrafo).
    2.  Si no cumple, buscar tamaños enteros divisores del bloque original, reduciendo secuencialmente: primero en dimensiones que no involucren una operación *Split-K*, para ver si la operación entra en memoria.
    3.  A cada tamaño $(w,h,k)$ candidato que cumpla con la restricción de Memoria Rápida, se le simula la latencia analítica de Cómputo vs. Memoria lenta.

*   **Gestión Asignación de Residencia:** Una clásica heurística *Belady's MIN* o *LRU (Least Recently Used)* adaptada servirá para decidir qué tensores (de `tensors_to_retain`) nos quedamos. Como conocemos el futuro del grafo, expulsamos (evict) a memoria lenta aquel tensor que se necesite lo más tarde posible en el futuro, manteniendo en Fast Memory a los de consumo inminente.
*   **Orden de Recorrido:** Cuando se aplica un *tiling spatial* menor al tamaño total, se aplica una heurística de ordenación simple: Si una entrada es compartida a través de filas (como MatMul RHS) o a través de columnas (MatMul LHS), se reordena iterando primero por esa dimensión repetida (ej. *Snake Order*) simulando así el caché a través de las iteraciones.

### Etapa 4: Simulador Analítico (Roofline Evaluator)

Escribiremos una función muy rápida parecida a `Evaluate()` de `mlsys.h`. Esta función calcula analíticamente la latencia basada en Cómputo vs IO de Transferencia para un determinado *schedule plan* (partición, tiles, residencia). La Etapa 2 y 3 usarán constantemente este simulador de forma iterativa y rápida para "Puntuar" qué tan buena es una hipótesis de Plan antes de generar el JSON final.

## 3. Estrategias de Optimización (Las tres tácticas ganadoras)

El sistema dará prioridad a implementar correctamente estas 3 tácticas avanzadas demostradas en la especificación:

1.  **Re-computación en vez de Spilling (The "Flash" Approach):** Si el motor detecta que el límite de Memoria Rápida obliga a escribir y volver a leer un tensor a la memoria lenta masiva, y dicho tensor fue barato computacionalmente (bajas operaciones aritméticas en Pointwise); resultará más barato recalcularlo desde sus entradas, fusionándolo dos veces en dos subgrafos distintos.
2.  **Split-K / Reducción Parcial:** Al detectar un `OpType = MatMul` en un ambiente con poca memoria. El algoritmo reducirá radicalmente el parámetro $K$ (ej a 1/4). El motor acumulará repetitivamente sobre el tensor de salida, transformando una operación gigante *limitada por la memoria (Memory Bound)* en varias porciones *limitadas por cómputo (Compute Bound)*.
3.  **Snake Traversal:** Por defecto, cualquier *tile / tileado* usará una curva-Z o Zig-zag en el índice espacial $w$ y $h$ en lugar del *Raster order* clásico, reduciendo masivamente el impacto de transferencias.

## 4. Tecnologías y Librerías C++ a Usar

1.  Lenguaje principal en **C++20** (para máxima velocidad y cumplimiento estricto de track A en Ubuntu 22.04 LTS).
2.  Usar la librería incluida `abseil-cpp` que ofrece el parser JSON eficiente y estructuras de datos estables (`StatusOr`).
3.  Emplear estructuras planas como vectores `std::vector` (evitando apuntadores dinámicos que ensucian caché L1 de CPU). Para árboles de búsqueda emplearemos bitsets (`std::bitset` de `<bit>`) para representar las máscaras de qué operaciones forman un sub-grafo de manera muy optimizada.

## Conclusión

El éxito de Track A no recae en probar Fuerza Bruta mediante código espagueti. Recae en construir un optimizador basado en una **búsqueda heurística iterativa**, guiada mediante el **Simulador Roofline interno** que premia matemáticamente "Mega Grupos" con Tiling Adaptable. Este equilibrio nos evitará colisiones `Out-of-memory` mientras nos mantiene asintóticamente cerca de los ciclos óptimos del Hardware para maximizar nuestros Puntos Ganadores.
