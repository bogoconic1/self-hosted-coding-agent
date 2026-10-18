# self-hosted-coding-agent

Claude Code running on a self-hosted model, with six local tools: `run_command`, `read_file`, `write_file`, `edit_file`, `get_status`, `submit_patch`.

Settings live in `config.yaml`.

## Setup

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
```

## Model (GPU host)

```bash
uv run --no-project --with pyyaml==6.0.3 python serve_qwen.py install --config config.yaml
uv run --no-project --with pyyaml==6.0.3 python serve_qwen.py serve --config config.yaml
```

Then tunnel it to your machine:

```bash
ssh -N -L 127.0.0.1:18000:127.0.0.1:8000 -p <port> root@<host>
```

## Run Claude Code

```bash
.venv/bin/python run_claude.py --config config.yaml <repo> <run-dir>
```

Wait until `/mcp` shows `ws` connected, then type your task. Add `--resume` to continue a session.

## Tools

```bash
.venv/bin/python tool_server.py --repo <repo> --run-dir <dir outside repo> --config config.yaml
```

The repo must start clean. `submit_patch` writes `<run-dir>/patch.diff`; apply it with `git apply`.

## Tests

```bash
.venv/bin/pytest tests/test_tools.py
```
