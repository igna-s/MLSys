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

# Build V17
WORKDIR /build/v17
COPY v17/main.cpp v17/scheduler.cpp v17/scheduler.h ./
COPY nlohmann/ ./nlohmann/
RUN g++ -O3 -std=c++20 -static -fopenmp -I. main.cpp scheduler.cpp -o mlsys && strip mlsys

# Build V18
WORKDIR /build/v18
COPY v18/main.cpp v18/scheduler.cpp v18/scheduler.h ./
COPY nlohmann/ ./nlohmann/
RUN g++ -O3 -std=c++20 -static -fopenmp -I. main.cpp scheduler.cpp -o mlsys && strip mlsys

# Build V19
WORKDIR /build/v19
COPY v19/main.cpp v19/scheduler.cpp v19/scheduler.h ./
COPY nlohmann/ ./nlohmann/
RUN g++ -O3 -std=c++20 -static -fopenmp -I. main.cpp scheduler.cpp -o mlsys && strip mlsys

# Build V20
WORKDIR /build/v20
COPY v20/main.cpp v20/scheduler.cpp v20/mlsys.h ./
COPY v20/third_party/ ./third_party/
COPY nlohmann/ ./nlohmann/
RUN g++ -O3 -std=c++20 -static -fopenmp -I. main.cpp scheduler.cpp -o mlsys && strip mlsys

# Build V21
WORKDIR /build/v21
COPY v21/main.cpp v21/scheduler.cpp v21/mlsys.h ./
COPY v21/third_party/ ./third_party/
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
COPY --from=builder /build/v17/mlsys /app/bin/mlsys_v17
COPY --from=builder /build/v18/mlsys /app/bin/mlsys_v18
COPY --from=builder /build/v19/mlsys /app/bin/mlsys_v19
COPY --from=builder /build/v20/mlsys /app/bin/mlsys_v20
COPY --from=builder /build/v21/mlsys /app/bin/mlsys_v21

COPY benchmarks/ /app/benchmarks/
COPY evaluate.py /app/evaluate.py

ENTRYPOINT ["python3", "/app/evaluate.py"]
