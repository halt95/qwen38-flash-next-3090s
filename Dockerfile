# Flash-Next v2.0.1 in one image: Qwen3.8-Flash-Next-W4A16-Merlin at 262K per request, 806,792-token KV pool,
# 4x RTX 3090. The image runs scripts/build-v2.sh exactly as the clean-room reproduction did (pinned upstream
# commit + the v2.0.1 bundle, commit AND tree hash asserted, pinned venv, compiled ops from the pinned upstream
# wheel -- no compiler, no CUDA toolkit) and starts scripts/serve-v2.sh through scripts/docker-entrypoint.sh.
#
#   docker build -t qwen38-flash-next-3090s:v2.0.1 .
#   docker run --gpus all --shm-size=8g --ulimit memlock=-1 -p 8000:8000 \
#     -v /path/to/Qwen3.8-Flash-Next-W4A16-Merlin:/models/Qwen3.8-Flash-Next-W4A16-Merlin \
#     -v flash-next-cache:/cache qwen38-flash-next-3090s:v2.0.1
#
# The bundle (release asset, ~600 KB) is fetched from BUNDLE_URL at build time, or taken from the build context if
# you put v2.0.1-from-upstream-e962733e08.bundle next to this Dockerfile. Its sha256 is asserted by build-v2.sh.
# Host needs: the NVIDIA driver with peer-to-peer working across the four cards, nvidia-container-toolkit,
# host RAM: 96 GB is the qualified allocation; the measured resident floor is ~69 GiB (~48 GiB n-gram table + ~4.2 GiB
# pinned embeddings + ~16 GiB workers/engine/shared segments), the rest being reclaimable page cache. /dev/shm >= 1 GB (--shm-size).
FROM python:3.13-slim-bookworm
ARG BUNDLE_URL=https://github.com/halt95/qwen38-flash-next-3090s/releases/download/v2.0.1/v2.0.1-from-upstream-e962733e08.bundle.gz
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates tar gzip procps \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/flash-next
# the repo files build-v2.sh / serve-v2.sh need (see .dockerignore for what is left out)
COPY . /opt/flash-next/
RUN if [ ! -f v2.0.1-from-upstream-e962733e08.bundle ]; then \
      curl -fsSL "$BUNDLE_URL" | gunzip > v2.0.1-from-upstream-e962733e08.bundle; fi \
    && chmod +x scripts/*.sh scripts/make-e1-config.py \
    && BUNDLE=/opt/flash-next/v2.0.1-from-upstream-e962733e08.bundle bash scripts/build-v2.sh /opt/vllm-v2 /opt/venv-v2 \
    && rm -rf /opt/vllm-v2/.wheel /root/.cache
ENV TREE=/opt/vllm-v2 VENV=/opt/venv-v2 HOST=0.0.0.0 PORT=8000 CACHE_ROOT=/cache
VOLUME /cache
EXPOSE 8000
# the first boot compiles the graphs (~10 min); later boots reuse /cache
# A real one-token generation, not just /v1/models: that endpoint answers 200 as soon as the HTTP server is
# listening and keeps answering after the engine has died, so it cannot distinguish healthy from dead. The
# timeout is generous because a health probe can land behind a queued 262K prefill.
HEALTHCHECK --interval=60s --timeout=90s --start-period=30m --retries=3 \
  CMD m="${MODEL_NAME%% *}"; m="${m:-flash-next-v2}"; curl -fs -m 85 -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"$m\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_tokens\":1,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
      | grep -q '"choices"' || exit 1
ENTRYPOINT ["/opt/flash-next/scripts/docker-entrypoint.sh"]
CMD ["/models/Qwen3.8-Flash-Next-W4A16-Merlin"]
