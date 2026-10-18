# self-hosted-coding-agent

Claude Code running on a self-hosted model, with six local tools: `run_command`, `read_file`, `write_file`, `edit_file`, `get_status`, `submit_patch`.

Settings live in `config.yaml`.

## Setup

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
```

## Tools

```bash
.venv/bin/python tool_server.py --repo <repo> --run-dir <dir outside repo> --config config.yaml
```

The repo must start clean. `submit_patch` writes `<run-dir>/patch.diff`; apply it with `git apply`.

## Tests

```bash
.venv/bin/pytest tests/test_tools.py
```
