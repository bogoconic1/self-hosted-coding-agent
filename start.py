# /// script
# requires-python = ">=3.12"
# dependencies = ["pyyaml==6.0.3", "python-dotenv==1.2.3"]
# ///
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config.yaml"
ENV = ROOT / ".env"
POLL_SECONDS = 5
UNSET = [
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_MODEL",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDE_CODE_USE_MANTLE",
]
MODEL_ALIASES = [
    "ANTHROPIC_DEFAULT_FABLE_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "CLAUDE_CODE_SUBAGENT_MODEL",
]


@dataclass(frozen=True)
class GpuHost:
    ssh: str
    port: int
    identity: str


@dataclass(frozen=True)
class Vllm:
    version: str
    port: int
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
    gpu_host: GpuHost
    vllm: Vllm
    claude: Claude

    @property
    def session(self) -> str:
        return f"vllm-{self.vllm.served_name}"

    @property
    def log(self) -> str:
        return f"$HOME/{self.session}.log"


def load() -> Config:
    if not ENV.exists():
        sys.exit(f"{ENV} is missing; copy .env.example to .env and fill it in")
    secrets = dotenv_values(ENV)
    raw = yaml.safe_load(CONFIG.read_text())
    config = Config(
        gpu_host=GpuHost(
            ssh=secrets["GPU_SSH"],
            port=int(secrets["GPU_SSH_PORT"]),
            identity=secrets["GPU_SSH_IDENTITY"],
        ),
        vllm=Vllm(**raw["vllm"]),
        claude=Claude(**raw["claude"]),
    )
    if {"host", "port"} & config.vllm.serve_args.keys():
        sys.exit("set the port with vllm.port; host and port are not serve_args")
    return config


def ssh(host: GpuHost, *options: str) -> list[str]:
    return [
        *("ssh", "-F", "/dev/null", "-i", os.path.expanduser(host.identity)),
        *("-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "LogLevel=ERROR"),
        *("-p", str(host.port)),
        *options,
        host.ssh,
    ]


def remote(host: GpuHost, command: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [*ssh(host), command], check=False, capture_output=True, text=True
    )


def serve_flags(serve_args: dict) -> list[str]:
    flags = []
    for name, value in serve_args.items():
        flags += [f"--{name}"] if value is True else [f"--{name}", str(value)]
    return flags


def start_server(config: Config) -> None:
    host = config.gpu_host
    if remote(host, f"tmux has-session -t {config.session}").returncode == 0:
        stop = shlex.join([*ssh(host), f"tmux kill-session -t {config.session}"])
        sys.exit(
            f"{config.session} is already running on the GPU host; stop it with:\n{stop}"
        )
    vllm = config.vllm
    serve = shlex.join(
        [
            *("uv", "run", "--no-project", "--with", f"vllm=={vllm.version}"),
            *("vllm", "serve", vllm.model),
            *("--host", "127.0.0.1", "--port", str(vllm.port)),
            *serve_flags(vllm.serve_args),
        ]
    )
    command = f"{serve} 2>&1 | tee {config.log}"
    subprocess.run(
        [*ssh(host), f"tmux new -d -s {config.session} {shlex.quote(command)}"],
        check=True,
    )


def open_tunnel(config: Config) -> subprocess.Popen:
    forward = f"127.0.0.1:{config.vllm.port}:127.0.0.1:{config.vllm.port}"
    return subprocess.Popen(
        ssh(
            config.gpu_host,
            *("-N", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30"),
            *("-L", forward),
        ),
        start_new_session=True,
    )


def wait_until_healthy(config: Config, tunnel: subprocess.Popen) -> None:
    health = f"http://127.0.0.1:{config.vllm.port}/health"
    while True:
        if tunnel.poll() is not None:
            sys.exit(f"tunnel exited; is local port {config.vllm.port} already in use?")
        status = remote(
            config.gpu_host,
            f"tmux has-session -t {config.session} && tail -n 1 {config.log}",
        )
        if status.returncode:
            sys.exit(
                f"vLLM exited before it became healthy; see {config.log} on the GPU host"
            )
        try:
            with urllib.request.urlopen(health, timeout=POLL_SECONDS):
                return
        except (urllib.error.URLError, ConnectionError):
            width = shutil.get_terminal_size().columns - 1
            print(f"\r{status.stdout.strip()[:width]:<{width}}", end="", flush=True)
            time.sleep(POLL_SECONDS)


def claude_command(config: Config, claude_args: list[str]) -> list[str]:
    return [
        *("claude", "--model", config.vllm.served_name),
        *(["--bare"] if config.claude.bare else []),
        *("--permission-mode", config.claude.permission_mode),
        *claude_args,
    ]


def claude_env(config: Config) -> dict[str, str]:
    credential = "ANTHROPIC_API_KEY" if config.claude.bare else "ANTHROPIC_AUTH_TOKEN"
    env = {key: value for key, value in os.environ.items() if key not in UNSET}
    return env | {
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{config.vllm.port}",
        credential: "local-only",
        **dict.fromkeys(MODEL_ALIASES, config.vllm.served_name),
        "CLAUDE_CODE_EFFORT_LEVEL": config.claude.effort,
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(config.vllm.max_model_len),
    }


def run(config: Config, claude_args: list[str]) -> int:
    print(f"starting {config.vllm.model} as {config.session} on the GPU host")
    start_server(config)
    tunnel = open_tunnel(config)
    try:
        wait_until_healthy(config, tunnel)
        print(f"\n{config.vllm.served_name} is ready; starting Claude Code")
        signal.signal(signal.SIGINT, lambda signum, frame: None)
        return subprocess.run(
            claude_command(config, claude_args),
            check=False,
            env=claude_env(config),
        ).returncode
    except KeyboardInterrupt:
        return 130
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print(f"\nstopping {config.session} and the tunnel")
        remote(config.gpu_host, f"tmux kill-session -t {config.session}")
        tunnel.terminate()
        tunnel.wait()


if __name__ == "__main__":
    sys.exit(run(load(), sys.argv[1:]))
