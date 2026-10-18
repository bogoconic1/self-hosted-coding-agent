import argparse
import json
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import yaml

from serve_qwen import Model
from tool_server import Limits

ROOT = Path(__file__).resolve().parent
UNSET = [
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_MODEL",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDE_CODE_USE_MANTLE",
]


@dataclass(frozen=True)
class ClaudeCode:
    local_port: int
    effort: str
    max_output_tokens: int
    mcp_tool_timeout_ms: int


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("repo", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("claude_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    claude = ClaudeCode(**config["claude_code"])
    model = Model(**config["model"])
    limits = Limits(**config["tools"])
    repo = args.repo.resolve()
    run_dir = args.run_dir.resolve()

    if run_dir.is_relative_to(repo):
        sys.exit(f"run dir {run_dir} is inside repository {repo}")
    if claude.max_output_tokens >= model.max_model_len:
        sys.exit("claude_code.max_output_tokens must be below model.max_model_len")
    if (
        claude.mcp_tool_timeout_ms
        <= (limits.command_timeout_s + limits.kill_grace_s) * 1000
    ):
        sys.exit(
            "claude_code.mcp_tool_timeout_ms must exceed command timeout + kill grace"
        )
    base_url = f"http://127.0.0.1:{claude.local_port}"
    served = json.load(urllib.request.urlopen(f"{base_url}/v1/models", timeout=5))
    if model.served_name not in [entry["id"] for entry in served["data"]]:
        sys.exit(f"{base_url} does not serve {model.served_name}")

    run_dir.mkdir(parents=True, exist_ok=True)
    mcp_config = run_dir / "mcp.json"
    mcp_config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "ws": {
                        "type": "stdio",
                        "command": sys.executable,
                        "args": [
                            str(ROOT / "tool_server.py"),
                            *("--repo", str(repo)),
                            *("--run-dir", str(run_dir)),
                            *("--config", str(config_path)),
                        ],
                        "timeout": claude.mcp_tool_timeout_ms,
                    }
                }
            },
            indent=2,
        )
    )

    env = {key: value for key, value in os.environ.items() if key not in UNSET}
    env |= {
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_API_KEY": "local-only",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model.served_name,
        "CLAUDE_CODE_EFFORT_LEVEL": claude.effort,
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(model.max_model_len),
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(claude.max_output_tokens),
        "ENABLE_TOOL_SEARCH": "false",
    }
    os.chdir(repo)
    os.execvpe(
        "claude",
        [
            *("claude", "--bare", "--model", model.served_name, "--tools", ""),
            *("--permission-mode", "default"),
            *("--strict-mcp-config", "--mcp-config", str(mcp_config)),
            *("--system-prompt-file", str(ROOT / "system_prompt.md")),
            *args.claude_args,
        ],
        env,
    )


if __name__ == "__main__":
    main()
