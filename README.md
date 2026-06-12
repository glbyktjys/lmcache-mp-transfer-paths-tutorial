# LMCache MP Mode — Non-CUDA Transfer Paths

Deployment guide and validation script for LMCache MP mode on CPU (non-CUDA SHM and pickle transfer paths).

📖 Blog: <link>

## Contents
- [`LMCache_DataTransfer.md`](LMCache_DataTransfer.md) —
step-by-step setup + run (SHM / pickle)
- [`cache_validation.sh`](cache_validation.sh) — end-to-end
cache-hit check across a vLLM restart