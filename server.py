import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import modal
import yaml

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
VLLM_PORT = 8000


@dataclass(frozen=True)
class ModalSettings:
    gpu: str
    startup_timeout_s: int
    idle_shutdown_s: int


@dataclass(frozen=True)
class Vllm:
    version: str
    model: str
    serve_args: dict

    @property
    def served_name(self) -> str:
        return self.serve_args["served-model-name"]

    @property
    def max_model_len(self) -> int:
        return self.serve_args["max-model-len"]


@dataclass(frozen=True)
class Claude:
    bare: bool
    permission_mode: str
    effort: str


@dataclass(frozen=True)
class Config:
    modal: ModalSettings
    vllm: Vllm
    claude: Claude


def load_config() -> Config:
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    config = Config(
        modal=ModalSettings(**raw["modal"]),
        vllm=Vllm(**raw["vllm"]),
        claude=Claude(**raw["claude"]),
    )
    if {"host", "port"} & config.vllm.serve_args.keys():
        sys.exit("host and port are set by server.py; remove them from serve_args")
    return config


def serve_flags(serve_args: dict) -> list[str]:
    flags = []
    for name, value in serve_args.items():
        flags += [f"--{name}"] if value is True else [f"--{name}", str(value)]
    return flags


CONFIG = load_config()

app = modal.App(f"vllm-{CONFIG.vllm.served_name}")
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.2-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(f"vllm=={CONFIG.vllm.version}", "pyyaml==6.0.3")
    .env({"TRITON_CACHE_DIR": "/root/.cache/triton"})
    .add_local_file(CONFIG_PATH, "/root/config.yaml")
)


@app.server(
    image=image,
    gpu=CONFIG.modal.gpu,
    scaledown_window=CONFIG.modal.idle_shutdown_s,
    startup_timeout=CONFIG.modal.startup_timeout_s,
    port=VLLM_PORT,
    volumes={
        "/root/.cache": modal.Volume.from_name(
            "vllm-server-cache", create_if_missing=True
        )
    },
)
class VllmServer:
    @modal.enter()
    def start(self) -> None:
        self.process = subprocess.Popen(
            [
                *("vllm", "serve", CONFIG.vllm.model),
                *("--host", "0.0.0.0", "--port", str(VLLM_PORT)),
                *serve_flags(CONFIG.vllm.serve_args),
            ]
        )

    @modal.exit()
    def stop(self) -> None:
        self.process.terminate()
        self.process.wait()
