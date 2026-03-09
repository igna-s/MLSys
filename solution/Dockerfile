# Multi-stage Dockerfile para compilar mlsys estáticamente en Ubuntu 22.04
# =========================================================================

FROM ubuntu:22.04 AS builder

# Evitar prompts interactivos durante apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Instalar herramientas de compilación
RUN apt-get update && apt-get install -y \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copiar código fuente
WORKDIR /src
COPY main.cpp scheduler.cpp scheduler.h ./
COPY nlohmann/ ./nlohmann/

# Compilar estáticamente con optimizaciones
RUN g++ -O3 -std=c++20 -static -fopenmp \
    -I. \
    main.cpp scheduler.cpp \
    -o mlsys \
    && strip mlsys

# Stage final: imagen limpia solo con el binario
FROM ubuntu:22.04

WORKDIR /app
COPY --from=builder /src/mlsys .

ENTRYPOINT ["./mlsys"]
