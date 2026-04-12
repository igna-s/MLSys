FROM ubuntu:22.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y g++ && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY main.cpp scheduler.cpp scheduler.h ./
COPY nlohmann/ ./nlohmann/
RUN g++ -O3 -std=c++20 -static -fopenmp -I. main.cpp scheduler.cpp -o mlsys && strip mlsys

FROM ubuntu:22.04
WORKDIR /app
COPY --from=builder /src/mlsys .
ENTRYPOINT ["./mlsys"]
