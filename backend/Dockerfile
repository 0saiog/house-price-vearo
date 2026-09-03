# Multi-stage: the toolchain and the 1.5 GB of build artifacts stay out of the
# shipped image, which ends up as a static-ish binary plus the model files.

FROM rust:1-slim-bookworm AS builder
WORKDIR /build

# Cargo needs git for the Vearo dependency.
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy manifests first so dependency compilation caches across source edits.
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY core/Cargo.toml core/
COPY ml/Cargo.toml ml/
COPY backend/Cargo.toml backend/
RUN mkdir -p core/src ml/src backend/src \
    && echo "" > core/src/lib.rs \
    && echo "fn main() {}" > ml/src/main.rs \
    && echo "fn main() {}" > backend/src/main.rs \
    && echo "" > backend/src/lib.rs \
    && cargo build --release -p backend \
    && rm -rf core/src ml/src backend/src

COPY core core
COPY ml ml
COPY backend backend
RUN touch core/src/lib.rs backend/src/main.rs && cargo build --release -p backend

FROM debian:bookworm-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app

COPY --from=builder /build/target/release/backend /usr/local/bin/backend
COPY models /app/models

USER app
ENV HOST=0.0.0.0 \
    PORT=8000 \
    MODEL_PATH=/app/models/house_price.ve \
    PREPROCESS_PATH=/app/models/preprocess.json \
    RUST_LOG=backend=info

EXPOSE 8000
CMD ["backend"]
