# Running LMCache MP Mode on CPU (non-CUDA path)

A step-by-step guide to running the LMCache multiprocess (MP) server with vLLM on **CPU** only environment, exercising the non-CUDA transfer paths (SHM and pickle).

## CUDA path

Use the official quick start: https://docs.lmcache.ai/getting_started/quickstart.html

## Non-CUDA path - set up

### 1. Create the virtual environment

```bash
cd ~/LMCache
uv venv --python 3.12 .venv-cpu
source .venv-cpu/bin/activate
```

### 2. Install vLLM (CPU build)

Install vLLM **first** so torch resolves to the CPU build (avoids pulling CUDA wheels).

**Option A — nightly wheel (daily build):**
```bash
uv pip install vllm \
  --extra-index-url https://wheels.vllm.ai/nightly/cpu \
  --index-strategy first-index \
  --torch-backend cpu
```

**Option B — pinned stable release** (use this if the nightly fails at vLLM startup with `unsupported kv_caches format ...`, i.e. the nightly drifted ahead of the connector or that another compatibility issue occurred.):

```bash
export VLLM_VERSION=$(curl -s https://api.github.com/repos/vllm-project/vllm/releases/latest | jq -r .tag_name | sed 's/^v//')

uv pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cpu-cp38-abi3-manylinux_2_35_x86_64.whl --torch-backend cpu
```

### 3. Install LMCache (CPU-only, editable)

```bash
uv pip install -r requirements/build.txt
NO_GPU_EXT=1 uv pip install -e . --no-build-isolation
```

## Non-CUDA path — run (SHM)

Start LMCache **first**, wait until it's healthy, then start vLLM.

### Terminal 1 — LMCache MP server (SHM = default, no `--shm-name`)

```bash
cd ~/LMCache && source .venv-cpu/bin/activate
lmcache server --l1-size-gb 2 --eviction-policy LRU --chunk-size 128
```

### Terminal 2 — vLLM on CPU
  
```bash
cd ~/LMCache && source .venv-cpu/bin/activate
VLLM_TARGET_DEVICE=cpu vllm serve facebook/opt-125m \
  --port 8000 \
  --dtype bfloat16 \
  --disable-hybrid-kv-cache-manager \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.3 \
  --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both"}'
```

### Terminal 3 — verify

```bash
curl -s localhost:8080/healthcheck
curl -s localhost:8000/v1/models | grep opt-125m
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m",
    "prompt": "The lighthouse keeper climbed the stairs and lit the lamp",
    "max_tokens": 16,
    "temperature": 0
  }'
# confirm Terminal 1 logs:  
# [2026-06-11 21:08:34,625] LMCache INFO: Using shm non-GPU transfer strategy (server_transfer.py:66:lmcache.v1.multiprocess.modules.server_transfer)
```

## Non-CUDA path — run (Pickle)

Terminals 2 and 3 are unchanged, but you must restart vLLM after starting a new LMCache server. vLLM registers with the LMCache server once, at startup — a new server instance doesn't know about an already-running vLLM, and requests will fail with:

LMCache ERROR: No GPU context found for model ... during lookup!

Restarting the LMCache server requires restarting vLLM. The reverse is fine — vLLM can restart freely against a running server.

### Terminal 1 — LMCache MP server (Pickle — `--shm-name ""`)

```bash
cd ~/LMCache && source .venv-cpu/bin/activate
lmcache server --l1-size-gb 2 --eviction-policy LRU --chunk-size 128 --shm-name ""
# confirm Terminal 1 logs:
# [2026-06-11 21:16:24,318] LMCache INFO: Using pickle non-GPU transfer strategy (server_transfer.py:76:lmcache.v1.multiprocess.modules.server_transfer)
```

## Cache Hit Validation

Beyond the manual steps above, `cache_validation.sh` runs the whole flow end to end: it starts its own LMCache MP server and vLLM worker, verifies stores and hits via the server's metrics, **restarts the vLLM worker**, and confirms the cache still hits across instances. Both processes are torn down on exit.

Prerequisites:

- The `.venv-cpu` environment from the setup section, activated. The script installs nothing — it runs `lmcache server` and `vllm serve` from the active environment.
- Ports 8000 and 8080 free — stop any LMCache/vLLM instances you started manually in the previous sections.

Run it with the transport path you want to test:

```bash
source ~/LMCache/.venv-cpu/bin/activate

bash ./cache_validation.sh shm       # SHM transport
bash ./cache_validation.sh pickle    # pickle transport
```

# If you are on macOS

Everything is the same except the vLLM install. Neither wheel option works on a Mac — the nightly index and the GitHub release both ship **Linux-only** manylinux wheels — so vLLM must be built from source ([official instructions](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/), Apple silicon tab).

## macOS - set up
Same flow as Linux: venv, then vLLM, then LMCache. Clone vLLM **next to** LMCache (not inside it); the active venv is what links the two installs, so `uv pip install` commands work from either directory.

> **Note — install order matters:** Always install vLLM before LMCache. LMCache's
> `NO_GPU_EXT=1 ... --no-build-isolation` install compiles C++ extensions via
> `torch.utils.cpp_extension`, so `torch` must already exist in the venv at build
> time — installing vLLM first provides it.

```bash
# --- Workspace, LMCache clone, and venv ---
mkdir demo && cd demo
git clone https://github.com/LMCache/LMCache.git
cd LMCache
uv venv --python 3.12 .venv-cpu
source .venv-cpu/bin/activate

# --- vLLM from source (also provides the torch LMCache's build needs) ---
cd ..
git clone https://github.com/vllm-project/vllm.git
cd vllm
uv pip install -r requirements/cpu.txt --index-strategy unsafe-best-match
uv pip install -e .

# --- LMCache (CPU-only, no GPU extensions) ---
cd ../LMCache
uv pip install -r requirements/build.txt
uv pip install cython   # macOS only: nvtx has no mac wheel; its sdist needs Cython
NO_GPU_EXT=1 uv pip install -e . --no-build-isolation
```

bf16 is not validated compute kernels for vLLM’s Apple Silicon CPU path. In this setup, it should be safe to try either --dtype float32 or --dtype float16, depending on what works best in your local environment.

## macOS — run

Two differences from the Linux commands:

- **dtype**: the macOS CPU backend supports fp32/fp16 only — `--dtype bfloat16`
  will not work. Use `--dtype float16` (opt-125m's checkpoint is fp16-native).
- **Ports**: development tools sometimes hold the default ports on a laptop
  (VS Code extensions are known to squat 5555 and 8080). If `lmcache server`
  fails with `ZMQError: Address already in use`, move both sides to a free
  port as shown below — the `--port` flag and the connector's
  `lmcache.mp.port` must match.

```bash
# Terminal 1
lmcache server --l1-size-gb 2 --eviction-policy LRU --chunk-size 128 --port 5556

# Terminal 2 
vllm serve facebook/opt-125m \
  --port 8000 \
  --dtype float16 \
  --disable-hybrid-kv-cache-manager \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.2 \
  --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_ex
tra_config":{"lmcache.mp.host":"tcp://localhost","lmcache.mp.port":5556}}'
```

## macOS — known issue

The LMCache side works on macOS: both transports (SHM and pickle) verifiably store chunks from the vLLM worker. But current vLLM nightlies have an upstream bug on Apple Silicon — the CPU attention kernel hangs on prompts longer than roughly one KV block (~100+ tokens). Short prompts work; the hang reproduces with bare vLLM, no LMCache involved.