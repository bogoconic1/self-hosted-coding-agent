# /// script
# requires-python = ">=3.12"
# dependencies = ["pyyaml==6.0.3", "python-dotenv==1.2.3", "modal==1.6.2"]
# ///
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import dotenv_values

ENV = Path(__file__).resolve().parent / ".env"
if not ENV.exists():
    sys.exit(f"{ENV} is missing; copy .env.example to .env and fill it in")
SECRETS = dotenv_values(ENV)
os.environ["MODAL_PROFILE"] = SECRETS["MODAL_PROFILE"]

from server import CONFIG, Config, VllmServer, app

POLL_SECONDS = 5
UNSET = [
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_CUSTOM_HEADERS",
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


def proxy_headers() -> dict[str, str]:
    return {
        "Modal-Key": SECRETS["MODAL_PROXY_KEY"],
        "Modal-Secret": SECRETS["MODAL_PROXY_SECRET"],
    }


def wait_until_healthy(url: str, headers: dict[str, str]) -> None:
    request = urllib.request.Request(f"{url}/health", headers=headers)
    started = time.monotonic()
    while True:
        try:
            with urllib.request.urlopen(request, timeout=POLL_SECONDS):
                return
        except urllib.error.HTTPError as error:
            if error.code == 401:
                sys.exit("Modal rejected the proxy token in .env")
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        elapsed = int(time.monotonic() - started)
        print(f"\rwaiting for vLLM on Modal: {elapsed}s", end="", flush=True)
        time.sleep(POLL_SECONDS)


def claude_command(config: Config, claude_args: list[str]) -> list[str]:
    return [
        *("claude", "--model", config.vllm.served_name),
        *(["--bare"] if config.claude.bare else []),
        *("--permission-mode", config.claude.permission_mode),
        *claude_args,
    ]


def claude_env(config: Config, url: str, headers: dict[str, str]) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key not in UNSET}
    return env | {
        "ANTHROPIC_BASE_URL": url,
        "ANTHROPIC_API_KEY": "local-only",
        "ANTHROPIC_CUSTOM_HEADERS": "\n".join(
            f"{name}: {value}" for name, value in headers.items()
        ),
        **dict.fromkeys(MODEL_ALIASES, config.vllm.served_name),
        "CLAUDE_CODE_EFFORT_LEVEL": config.claude.effort,
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(config.vllm.max_model_len),
    }


def run(config: Config, claude_args: list[str]) -> int:
    headers = proxy_headers()
    print(f"starting {config.vllm.model} on a Modal {config.modal.gpu}")
    with app.run():
        print(f"server log: modal app logs {app.app_id}")
        url = VllmServer.get_url()
        wait_until_healthy(url, headers)
        print(f"\n{config.vllm.served_name} is ready at {url}; starting Claude Code")
        signal.signal(signal.SIGINT, lambda signum, frame: None)
        return subprocess.run(
            claude_command(config, claude_args),
            check=False,
            env=claude_env(config, url, headers),
        ).returncode
    return 130


if __name__ == "__main__":
    sys.exit(run(CONFIG, sys.argv[1:]))
