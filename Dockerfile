# Flash-Next v2.2.0 in one image: Qwen3.8-Flash-Next-W4A16-Merlin at 262K per request, 806,792-token KV pool,
# 4x RTX 3090, on vLLM 0.30.0. The image runs scripts/build-v2.2.sh (public v0.30.0 + the v2.2.0 bundle, commit AND tree
# hash asserted, pinned venv, the two v2.2 extensions from the build-artifacts asset, hash-checked) and starts
# scripts/serve-v2.2.sh through scripts/docker-entrypoint.sh.
#
#   docker build -t qwen38-flash-next-3090s:v2.2.0 .
#   docker run --gpus all --stop-timeout 70 --shm-size=8g --ulimit memlock=-1 -p 8000:8000 \
#     -v /path/to/Qwen3.8-Flash-Next-W4A16-Merlin:/models/Qwen3.8-Flash-Next-W4A16-Merlin \
#     -v flash-next-cache-v2.2:/cache qwen38-flash-next-3090s:v2.2.0
#
# The two release assets are fetched at build time (BUNDLE_URL ~0.7 MB, ARTIFACTS_URL ~241 MB), or taken from the build
# context if you put v2.2.0-from-upstream-v0.30.0.bundle (or the downloaded .bundle.gz) and
# build-artifacts-sm86-py313-cu130-v2.2.0.tar.gz next to this Dockerfile. build-v2.2.sh asserts both sha256s from
# upstream/PIN-v2.2.
#
# Runtime kernel compilation. The build needs no compiler, but the FIRST SERVE does: FlashInfer compiles its prefill,
# sampling and top-k kernels and Triton compiles its launchers against Python.h. So the image carries a C/C++ compiler
# and ninja (the python:3.13 base image ships the Python headers), and the venv carries the CUDA 13.0
# nvcc/crt/nvvm/cccl wheels (PTX any CUDA >= 13.0 driver accepts; an unpinned toolkit resolves to a newer CUDA whose
# PTX an older driver rejects with "Unsupported .version"); build-v2.2.sh adds the lib64 / unversioned library links
# FlashInfer links against, and serve-v2.2.sh points CUDA_HOME at that toolkit and checks it against the driver
# before starting. (Reported on the v2.0.1 image by a user on a headless host.)
#
# Host needs: the NVIDIA driver (CUDA >= 13.0; the release is qualified with peer-to-peer working across the four cards, and
# also runs without it, with lower prefill: README, "Build and serve"), nvidia-container-toolkit registered with Docker
# (nvidia-ctk runtime configure --runtime=docker, then restart docker),
# host RAM: 96 GB is the qualified allocation; the measured resident floor is ~69 GiB (~48 GiB n-gram table + ~4.2 GiB
# pinned embeddings + ~16 GiB workers/engine/shared segments), the rest being reclaimable page cache. /dev/shm >= 1 GB (--shm-size).
FROM python:3.13-slim-bookworm
ARG BUNDLE_URL=https://github.com/halt95/qwen38-flash-next-3090s/releases/download/v2.2.0/v2.2.0-from-upstream-v0.30.0.bundle.gz
ARG ARTIFACTS_URL=https://github.com/halt95/qwen38-flash-next-3090s/releases/download/v2.2.0/build-artifacts-sm86-py313-cu130-v2.2.0.tar.gz
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates tar gzip procps \
       gcc g++ build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/flash-next
# the repo files build-v2.2.sh / serve-v2.2.sh need (see .dockerignore for what is left out)
COPY . /opt/flash-next/
# (a .bundle.gz from the build context is decompressed by build-v2.2.sh)
RUN if [ ! -f v2.2.0-from-upstream-v0.30.0.bundle ] && [ ! -f v2.2.0-from-upstream-v0.30.0.bundle.gz ]; then \
      curl -fsSL "$BUNDLE_URL" | gunzip > v2.2.0-from-upstream-v0.30.0.bundle; fi \
    && if [ ! -f build-artifacts-sm86-py313-cu130-v2.2.0.tar.gz ]; then \
      curl -fsSL -o build-artifacts-sm86-py313-cu130-v2.2.0.tar.gz "$ARTIFACTS_URL"; fi \
    && chmod +x scripts/*.sh scripts/make-e1-config.py \
    && BUNDLE=/opt/flash-next/v2.2.0-from-upstream-v0.30.0.bundle \
       ARTIFACTS=/opt/flash-next/build-artifacts-sm86-py313-cu130-v2.2.0.tar.gz \
       bash scripts/build-v2.2.sh /opt/vllm-v2.2 /opt/venv-v2.2 \
    && rm -f build-artifacts-sm86-py313-cu130-v2.2.0.tar.gz && rm -rf /root/.cache
ENV TREE=/opt/vllm-v2.2 VENV=/opt/venv-v2.2 HOST=0.0.0.0 PORT=8000 CACHE_ROOT=/cache \
    CUDA_HOME=/opt/venv-v2.2/lib/python3.13/site-packages/nvidia/cu13
ENV PATH="/opt/venv-v2.2/lib/python3.13/site-packages/nvidia/cu13/bin:$PATH"
# FlashInfer caches its JIT kernels under $FLASHINFER_WORKSPACE_BASE/.cache/flashinfer (default: the home directory,
# i.e. the container's writable layer, lost when the container is re-created); Triton's own cache defaults to
# ~/.triton/cache (vLLM moves it into the compile cache while it compiles). Both go to the /cache volume.
ENV FLASHINFER_WORKSPACE_BASE=/cache/flashinfer TRITON_CACHE_DIR=/cache/triton
VOLUME /cache
EXPOSE 8000
# the first boot compiles the graphs (~6 min), the first request FlashInfer's kernels; later boots reuse /cache
# A real one-token generation, not just /v1/models: that endpoint answers 200 as soon as the HTTP server is
# listening and keeps answering after the engine has died, so it cannot distinguish healthy from dead. The
# timeout is generous because a health probe can land behind a queued 262K prefill.
HEALTHCHECK --interval=60s --timeout=90s --start-period=30m --retries=3 \
  CMD m="${MODEL_NAME%% *}"; m="${m:-flash-next-v2}"; curl -fs -m 85 -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
      -H "Content-Type: application/json" ${VLLM_API_KEY:+-H "Authorization: Bearer $VLLM_API_KEY"} \
      -d "{\"model\":\"$m\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_tokens\":1,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
      | grep -q '"choices"' || exit 1
ENTRYPOINT ["/opt/flash-next/scripts/docker-entrypoint.sh"]
CMD ["/models/Qwen3.8-Flash-Next-W4A16-Merlin"]
