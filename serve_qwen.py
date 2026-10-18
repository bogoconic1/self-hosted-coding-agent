import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Model:
    checkpoint: str
    revision: str
    served_name: str
    port: int
    max_model_len: int
    max_num_seqs: int
    reasoning_parser: str
    tool_call_parser: str


@dataclass(frozen=True)
class Vllm:
    version: str
    venv: str


def install(vllm: Vllm) -> None:
    subprocess.run(["uv", "venv", vllm.venv, "--python", "3.12"], check=True)
    subprocess.run(
        [
            *("uv", "pip", "install"),
            *("--python", f"{vllm.venv}/bin/python"),
            f"vllm=={vllm.version}",
        ],
        check=True,
    )


def serve(model: Model, vllm: Vllm) -> None:
    installed = subprocess.run(
        [f"{vllm.venv}/bin/python", "-c", "import vllm; print(vllm.__version__)"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if installed != vllm.version:
        sys.exit(f"{vllm.venv} has vllm {installed}, config.yaml pins {vllm.version}")
    executable = f"{vllm.venv}/bin/vllm"
    os.execv(
        executable,
        [
            *(executable, "serve", model.checkpoint),
            *("--revision", model.revision),
            *("--served-model-name", model.served_name),
            *("--host", "127.0.0.1", "--port", str(model.port)),
            *("--max-model-len", str(model.max_model_len)),
            *("--max-num-seqs", str(model.max_num_seqs)),
            *("--reasoning-parser", model.reasoning_parser),
            "--enable-auto-tool-choice",
            *("--tool-call-parser", model.tool_call_parser),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["install", "serve"])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    model = Model(**config["model"])
    vllm = Vllm(**config["vllm"])
    if args.command == "install":
        install(vllm)
    else:
        serve(model, vllm)


if __name__ == "__main__":
    main()
