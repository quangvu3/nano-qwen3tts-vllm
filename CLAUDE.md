# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

nano-qwen3tts-vllm is a from-scratch, nano-vllm-style serving engine for Qwen3-TTS: continuous batching, paged KV cache, and CUDA graphs, aiming to match vLLM-level throughput/RTF in a small (~1k line core) codebase. It is under active development and mid-refactor — see "Known gap" below before trusting README usage snippets.

## Setup / running

```bash
uv sync            # or: pip install -e .
```

- Requires Python ≥3.10, PyTorch ≥2.10 with CUDA, compute capability ≥8.0 (Flash Attention). See README for the flash-attn prebuilt-wheel install command.
- **Package name quirk**: the importable module is `nano_qwen3tts_vllm` (underscore), but the source lives on disk at `nano-qwen3tts-vllm/` (hyphen — same name as the repo root itself, so there are two nested dirs with that name). The mapping is done via `[tool.setuptools.package-dir]` in `pyproject.toml`. When navigating the tree, the inner `nano-qwen3tts-vllm/` is the actual package.
- **No test suite or linter is configured.** `examples/test_interface_zmq.py` and `examples/quick_benchmark.py` are manual/interactive scripts, not pytest targets — running them requires a real model checkpoint and a GPU.
- Run the server:
  ```bash
  export QWEN3_TTS_MODEL_PATH=/path/to/model   # or a HF model id
  python examples/server.py
  # or: uvicorn examples.server:app --host 0.0.0.0 --port 8000
  ```
  Tunable env vars (`PREFILL_COLLECT_MS`, `PREDICTOR_COLLECT_MS`, `STREAMING_CHUNK_SIZE`, `DECODER_MP_WORKER`, etc.) are documented in the docstring at the top of `examples/server.py` — check there rather than duplicating.

## Known gap: README examples vs. actual behavior

The README shows synchronous usage (`interface.generate_custom_voice(...)`, `interface.generate_voice_design(...)`) as if they return generators you can `list(...)`. In the current code these methods immediately `raise RuntimeError` (they're generator bodies with the raise before any real work, kept only as documentation of the old single-process call shape). The only working path today is the **async multiprocess/ZMQ API**:

```python
await interface.start_zmq_tasks()
async for chunk in interface.generate_custom_voice_async(text=..., language=..., speaker=...):
    ...
```

Treat any new sync (non-`_async`) generation method the same way — it's dead/legacy unless proven otherwise.

## Architecture

**Two-model pipeline per audio frame.** A *Talker* LLM autoregressively emits one codec token (codebook 0) per step. That token's hidden state + embedding are fed to a *Predictor* LLM, which fills in the remaining 15 codebooks in a single extra forward pass. The Predictor's outputs are re-embedded and summed back into the next Talker input. This ping-pong is orchestrated in `interface.py`'s `generate_async` (the authoritative implementation; `_generate_caller_driven` is the old single-process version and now just raises).

**Engine layer** (`nano-qwen3tts-vllm/engine/`) mirrors nano-vllm's design:
- `llm_engine/base.py` (`LLMEngine`) is subclassed by `talker_llm_engine.py` and `predictor_llm_engine.py`.
- `model_runner/base.py` is subclassed by `talker_mode_runner.py` and `predictor_model_runner.py`.
- `scheduler.py` + `block_manager.py` + `sequence.py` implement continuous batching and paged KV-cache block allocation.
- `llm.py` just re-exports `TalkerLLM` / `PredictorLLM` as thin aliases over the engine classes.

**Multiprocess + ZMQ is the only supported runtime path.** Talker and Predictor each run in their own spawned process (`workers/talker_worker.py`, `workers/predictor_worker.py`). The main process only loads embedding/projection layers (`utils/embedding_loader.py`) — it never holds the full Talker/Predictor weights. Commands flow main → worker over ZMQ PUSH/PULL:
- `workers/client_bridge.py` binds the sockets, spawns the worker processes, and runs a background "result-bridge" thread that resolves asyncio Futures when worker replies arrive.
- `workers/protocol.py` (de)serializes commands/results (pickle + numpy): `add_request`, `run_step`, `clear_request`, `shutdown`.
- `zmq/engine_loop_mp.py` runs the per-engine asyncio orchestration loop in the main process and fans results out to per-request `asyncio.Queue`s that the async generators in `interface.py` consume.

**Memory management.** `interface.py::_compute_memory_split` takes one user-supplied `gpu_memory_utilization` fraction and auto-splits it between the Talker and Predictor processes, estimating weight size and per-block KV-cache size from the model config. It sets `process_gpu_memory_fraction` so each process only sees its own slice of VRAM — this is what lets multiple independent server instances coexist on one GPU.

**Top-level API** (`interface.py::Qwen3TTSInterface`): `from_pretrained` (HF download or local path) / `__init__`, `create_voice_clone_prompt`, `generate_custom_voice_async`, `generate_voice_clone_async`, `start_zmq_tasks` / `stop_zmq_tasks`, `shutdown`. All generation methods yield raw codec-chunk lists; decode audio via `interface.speech_tokenizer.decode(...)` (`utils/audio.py`, `utils/speech_tokenizer_cudagraph.py`).

**Three model families**, same engine/worker code path, different prompt-prep helpers in `utils/prompt.py` and `utils/generation.py` (`prepare_custom_voice_prompt`, `prepare_inputs`, `generate_speaker_prompt`, `generate_icl_prompt`):
- CustomVoice — pre-defined speaker timbres.
- VoiceDesign — natural-language voice description.
- Base — voice cloning from reference audio (ICL mode or x-vector-only mode).

`examples/server.py` is the reference FastAPI integration: batched/windowed streaming decode, `POST /v1/audio/speech` returning a `StreamingResponse` of 16-bit PCM.
