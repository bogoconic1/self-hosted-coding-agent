# self-hosted-coding-agent

Run Claude Code on a model you serve with vLLM on your own GPU host, with one command.

## Setup

Copy `.env.example` to `.env` and fill in your GPU host's SSH details. Set the model in `config.yaml`.

## Start

From the project you want to work on:

```bash
uv run ~/Documents/self-hosted-coding-agent/start.py
```

This starts vLLM on the GPU host, opens an SSH tunnel and opens Claude Code once the model is ready. Exit Claude Code (or press Ctrl-C while the model loads) to stop vLLM and the tunnel. Arguments are passed to `claude`, e.g. `start.py --resume`.
