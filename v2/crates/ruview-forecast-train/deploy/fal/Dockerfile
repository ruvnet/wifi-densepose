FROM nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:4b9ed5fa8361736996499f64ecebf25d4ec37ff56e4d11323ccde10aa36e0c43 AS build
ARG RUST_VERSION=1.92.0
WORKDIR /build
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain "${RUST_VERSION}"
ENV PATH="/root/.cargo/bin:${PATH}"
COPY .ruview-build-tree /build/.ruview-build-tree
COPY v2 ./v2
RUN test "$(wc -c < /build/.ruview-build-tree)" -le 128 \
    && RUVIEW_TREE="$(tr -d '\r\n' < /build/.ruview-build-tree)" \
    && RUVIEW_LOCK="$(sha256sum /build/v2/Cargo.lock | cut -d' ' -f1)" \
    && case "${RUVIEW_TREE}${RUVIEW_LOCK}" in *[!0-9a-f]*) exit 1;; esac \
    && RUVIEW_WORKER_BUILD_ID="ruview-${RUVIEW_TREE}" \
       RUVIEW_BUILD_MANIFEST_SHA256="${RUVIEW_LOCK}" \
       cargo build --manifest-path v2/Cargo.toml --locked --release \
         -p ruview-forecast-train --no-default-features --features cli,server,cuda

FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04@sha256:828c4d878adcaa4265d80c95d8ec877149b49bb2419a4cf3bb6aa889bbb7ca2e
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates python3 python3-pip \
    && python3 -m pip install --break-system-packages --no-cache-dir "fal==1.80.0" \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --create-home ruforecast \
    && mkdir -p /data/ruview-forecast/artifacts \
    && chown -R ruforecast:ruforecast /data/ruview-forecast
COPY --from=build /build/v2/target/release/ruforecast /usr/local/bin/ruforecast
USER ruforecast
EXPOSE 8000

# No ENTRYPOINT or CMD: fal's Python Direct Server launcher owns process
# startup and invokes the typed Rust `serve` command explicitly.
