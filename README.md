# self-hosted-coding-agent

Run Claude Code on a model you serve with vLLM on a Modal GPU, with one command.

## Setup

```bash
uvx modal setup
uvx modal profile list
uvx modal workspace proxy-tokens create --profile <your profile>
```

Copy `.env.example` to `.env` and fill in your Modal profile and the proxy token. Set the model and GPU in `config.yaml`.

## Start

From the project you want to work on:

```bash
uv run ~/Documents/self-hosted-coding-agent/start.py
```

This starts vLLM on Modal and opens Claude Code once the model is ready. Exit Claude Code (or press Ctrl-C while the model loads) to stop the Modal app. Arguments are passed to `claude`, e.g. `start.py --resume`.
