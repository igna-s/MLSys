FROM ubuntu:22.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y g++ && rm -rf /var/lib/apt/lists/*

# Build V10
WORKDIR /build/v10
COPY v10/main.cpp v10/scheduler.cpp v10/scheduler.h ./
COPY nlohmann/ ./nlohmann/
RUN g++ -O3 -std=c++20 -static -fopenmp -I. main.cpp scheduler.cpp -o mlsys && strip mlsys

# Build V14
WORKDIR /build/v14
COPY v14/main.cpp v14/scheduler.cpp v14/scheduler.h ./
COPY nlohmann/ ./nlohmann/
RUN g++ -O3 -std=c++20 -static -fopenmp -I. main.cpp scheduler.cpp -o mlsys && strip mlsys

# Build V15
WORKDIR /build/v15
COPY v15/main.cpp v15/scheduler.cpp v15/scheduler.h ./
COPY nlohmann/ ./nlohmann/
RUN g++ -O3 -std=c++20 -static -fopenmp -I. main.cpp scheduler.cpp -o mlsys && strip mlsys

# Build V16
WORKDIR /build/v16
COPY v16/main.cpp v16/scheduler.cpp v16/scheduler.h ./
COPY nlohmann/ ./nlohmann/
RUN g++ -O3 -std=c++20 -static -fopenmp -I. main.cpp scheduler.cpp -o mlsys && strip mlsys

# Final image with all binaries + benchmarks + evaluator
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y python3 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY --from=builder /build/v10/mlsys /app/bin/mlsys_v10
COPY --from=builder /build/v14/mlsys /app/bin/mlsys_v14
COPY --from=builder /build/v15/mlsys /app/bin/mlsys_v15
COPY --from=builder /build/v16/mlsys /app/bin/mlsys_v16

COPY benchmarks/ /app/benchmarks/
COPY evaluate.py /app/evaluate.py

ENTRYPOINT ["python3", "/app/evaluate.py"]
