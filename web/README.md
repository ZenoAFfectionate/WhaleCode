# WhaleCode Web Console

This directory contains a lightweight web console that controls the existing
WhaleCode agent infrastructure instead of serving a static showcase only.

## Run

```bash
cd /home/kemove/CodeingAgent/Whale_Code
python3 web/server.py --host 127.0.0.1 --port 8765
```

Then open:

```text
http://127.0.0.1:8765
```

## Environment

The web server reuses the same environment variables as the CLI:

- `LLM_MODEL_ID`
- `LLM_API_KEY`
- `LLM_BASE_URL`
- `CODE_AGENT_MAX_STEPS`

Optional vLLM process command overrides:

- `WHALE_WEB_VLLM_COMMAND`
- `WHALE_WEB_VLLM_COMMAND_QWEN35_35B_FP8`
- `WHALE_WEB_VLLM_COMMAND_QWEN3_CODER_30B`
- `WHALE_WEB_VLLM_COMMAND_DEEPSEEK_CODER_LITE`

If no override is set, the server uses the default command embedded in
`web/server.py` for the selected catalog model.

## API Surface

- `POST /api/agent/runs` creates an agent job.
- `GET /api/jobs/{job_id}/events` streams job events over SSE.
- `GET /api/sessions` lists web-created persisted sessions.
- `DELETE /api/sessions/{filename}` deletes a web session file.
- `GET /api/models` returns model/vLLM/GPU status.
- `POST /api/models/start` starts or switches vLLM.
- `POST /api/models/stop` stops vLLM.
- `POST /api/models/unload` stops vLLM and clears active model state.
- `GET /api/datasets` lists benchmark datasets.
- `POST /api/benchmarks/runs` launches benchmark scripts.
- `GET /api/benchmarks/history` lists result artifacts in `data/_results`.
