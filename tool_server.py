import argparse
import difflib
import functools
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import anyio
import yaml
from mcp.server import MCPServer
from mcp.types import CallToolResult, TextContent
from pydantic import Field

mcp = MCPServer("ws")


@dataclass(frozen=True)
class Limits:
    command_timeout_s: float
    kill_grace_s: float
    stream_chars: int
    read_lines: int
    read_chars: int
    diff_chars: int


@dataclass
class Run:
    repo: Path
    run_dir: Path
    limits: Limits
    state: dict
    lock: anyio.Lock = field(default_factory=anyio.Lock)
    pgid: int | None = None


RUN: Run


def git(repo: Path, *args: str, index: Path | None = None) -> bytes:
    env = os.environ | {"GIT_OPTIONAL_LOCKS": "0"}
    if index is not None:
        env["GIT_INDEX_FILE"] = str(index)
    result = subprocess.run(
        ["git", *args], check=False, cwd=repo, env=env, capture_output=True
    )
    if result.returncode:
        message = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"git {args[0]} failed: {message}")
    return result.stdout


def atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.ws-tmp")
    temporary.write_bytes(data)
    if path.exists():
        shutil.copymode(path, temporary)
    os.replace(temporary, path)


def save_state() -> None:
    atomic_write(RUN.run_dir / "run.json", json.dumps(RUN.state, indent=2).encode())


def open_run(argv: list[str]) -> Run:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    limits = Limits(**yaml.safe_load(args.config.read_text())["tools"])
    repo = args.repo.resolve()
    run_dir = args.run_dir.resolve()
    if run_dir.is_relative_to(repo):
        sys.exit(f"run dir {run_dir} is inside repository {repo}")
    if git(repo, "rev-parse", "--show-toplevel").decode().strip() != str(repo):
        sys.exit(f"{repo} is not the root of a Git repository")
    head = git(repo, "rev-parse", "--verify", "HEAD").decode().strip()
    state_path = run_dir / "run.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["repo"] != str(repo):
            sys.exit(f"{state_path} belongs to {state['repo']}, not {repo}")
    elif git(repo, "status", "--porcelain"):
        sys.exit(f"working tree of {repo} is not clean")
    else:
        state = {
            "repo": str(repo),
            "baseline_commit": head,
            "started_at": time.time(),
            "tool_calls": 0,
            "last_patch": None,
        }
    (run_dir / "commands").mkdir(parents=True, exist_ok=True)
    atomic_write(state_path, json.dumps(state, indent=2).encode())
    return Run(repo=repo, run_dir=run_dir, limits=limits, state=state)


def tool(fn):
    @functools.wraps(fn)
    async def call(**kwargs) -> CallToolResult:
        async with RUN.lock:
            RUN.state["tool_calls"] += 1
            started = time.monotonic()
            try:
                data = await fn(**kwargs)
                is_error = False
            except (OSError, ValueError, RuntimeError) as error:
                data = {"type": type(error).__name__, "message": str(error)}
                is_error = True
            finally:
                save_state()
            event = {
                "ts": time.time(),
                "tool": fn.__name__,
                "args": kwargs,
                "is_error": is_error,
                "result": data,
                "duration_s": round(time.monotonic() - started, 3),
            }
            with (RUN.run_dir / "tools.jsonl").open("a") as log:
                log.write(json.dumps(event) + "\n")
            text = json.dumps(data)
            return CallToolResult(
                content=[TextContent(type="text", text=text)], is_error=is_error
            )

    return mcp.tool(structured_output=False)(call)


def resolve(filepath: str) -> Path:
    path = (RUN.repo / filepath).resolve()
    if (
        Path(filepath).is_absolute()
        or path == RUN.repo
        or not path.is_relative_to(RUN.repo)
    ):
        raise PermissionError(
            f"{filepath}: must be a relative path inside the repository"
        )
    if ".git" in (part.lower() for part in path.relative_to(RUN.repo).parts):
        raise PermissionError(f"{filepath}: .git is not writable or readable here")
    return path


def killpg(pgid: int, sig: signal.Signals) -> None:
    with suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, sig)


def excerpt(path: Path) -> tuple[str, bool]:
    limit = RUN.limits.stream_chars
    half = limit // 2
    with path.open("rb") as stream:
        head = stream.read(4 * limit + 4).decode(errors="replace")
        if len(head) <= limit:
            return head, False
        end = stream.seek(0, os.SEEK_END)
        stream.seek(max(end - 4 * half, 0))
        tail = stream.read().decode(errors="replace")
    return f"{head[:half]}\n[... truncated ...]\n{tail[-half:]}", True


def changes() -> tuple[bytes, list[str]]:
    index = RUN.run_dir / "patch.index"
    baseline = RUN.state["baseline_commit"]
    try:
        git(RUN.repo, "read-tree", baseline, index=index)
        git(RUN.repo, "add", "-A", "--", ".", index=index)
        patch = git(
            RUN.repo,
            *("diff-index", "--cached", "-p", "--binary", "--full-index"),
            *(baseline, "--", "."),
            index=index,
        )
        names = git(
            RUN.repo,
            *("diff-index", "--cached", "--name-only", "-z", baseline, "--", "."),
            index=index,
        )
    finally:
        index.unlink(missing_ok=True)
    return patch, names.decode().split("\0")[:-1]


@tool
async def run_command(
    command: Annotated[
        str,
        Field(
            description="Shell command, run with /bin/bash -c at the repository root."
        ),
    ],
) -> dict:
    """Run a shell command in the repository with a timeout. Each call starts a fresh shell. Long streams keep their beginning and end; the full output is saved at the returned paths."""
    number = RUN.state["tool_calls"]
    stdout_path = RUN.run_dir / "commands" / f"{number}.stdout"
    stderr_path = RUN.run_dir / "commands" / f"{number}.stderr"
    started = time.monotonic()
    timed_out = False
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        async with await anyio.open_process(
            ["/bin/bash", "-c", command],
            cwd=RUN.repo,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        ) as process:
            RUN.pgid = process.pid
            try:
                with anyio.move_on_after(RUN.limits.command_timeout_s) as deadline:
                    await process.wait()
                if deadline.cancelled_caught:
                    timed_out = True
                    killpg(process.pid, signal.SIGTERM)
                    with anyio.move_on_after(RUN.limits.kill_grace_s):
                        await process.wait()
            finally:
                killpg(process.pid, signal.SIGKILL)
                RUN.pgid = None
    stdout_text, stdout_truncated = excerpt(stdout_path)
    stderr_text, stderr_truncated = excerpt(stderr_path)
    return {
        "exit_code": None if timed_out else process.returncode,
        "timed_out": timed_out,
        "duration_s": round(time.monotonic() - started, 3),
        "stdout": stdout_text,
        "stderr": stderr_text,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }


@tool
async def read_file(
    filepath: Annotated[
        str, Field(description="Path relative to the repository root.")
    ],
    start_line: Annotated[int, Field(description="First line, 1-based.")] = 1,
    end_line: Annotated[
        int | None, Field(description="Last line, inclusive. Defaults to end of file.")
    ] = None,
) -> dict:
    """Read a UTF-8 text file by 1-based inclusive line range. Each call returns a bounded number of lines and characters; check truncated and request the next range. Content has no line numbers and keeps the file's line endings."""
    path = resolve(filepath)
    with path.open(encoding="utf-8", newline="") as stream:
        lines = stream.read().split("\n")
    if lines[-1] == "":
        lines.pop()
    total = len(lines)
    if start_line < 1 or start_line > max(total, 1):
        raise ValueError(f"start_line {start_line} is outside 1..{max(total, 1)}")
    if end_line is not None and end_line < start_line:
        raise ValueError(f"end_line {end_line} is before start_line {start_line}")
    last = total if end_line is None else min(end_line, total)
    end = min(last, start_line + RUN.limits.read_lines - 1)
    content = "\n".join(lines[start_line - 1 : end])
    return {
        "filepath": filepath,
        "start_line": start_line,
        "end_line": end,
        "total_lines": total,
        "truncated": end < last or len(content) > RUN.limits.read_chars,
        "content": content[: RUN.limits.read_chars],
    }


@tool
async def write_file(
    filepath: Annotated[
        str, Field(description="Path relative to the repository root.")
    ],
    content: Annotated[str, Field(description="Full new content of the file.")],
) -> dict:
    """Create or overwrite a UTF-8 text file, creating parent directories as needed."""
    path = resolve(filepath)
    if path.is_dir():
        raise IsADirectoryError(f"{filepath} is a directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = content.encode()
    atomic_write(path, data)
    return {"filepath": filepath, "bytes_written": len(data)}


@tool
async def edit_file(
    filepath: Annotated[
        str, Field(description="Path relative to the repository root.")
    ],
    old_string: Annotated[
        str, Field(description="Exact text to replace, including whitespace.")
    ],
    new_string: Annotated[str, Field(description="Replacement text.")],
    allow_multiple: Annotated[
        bool, Field(description="Replace every occurrence instead of exactly one.")
    ] = False,
) -> dict:
    """Replace an exact, unique occurrence of old_string in a file. Include enough surrounding lines to make old_string unique, or set allow_multiple."""
    path = resolve(filepath)
    if not old_string:
        raise ValueError("old_string is empty")
    with path.open(encoding="utf-8", newline="") as stream:
        text = stream.read()
    count = text.count(old_string)
    if count == 0:
        raise ValueError(f"old_string not found in {filepath}")
    if count > 1 and not allow_multiple:
        raise ValueError(
            f"old_string occurs {count} times in {filepath}; add surrounding context"
        )
    updated = text.replace(old_string, new_string)
    atomic_write(path, updated.encode())
    diff = "".join(
        difflib.unified_diff(
            text.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            f"a/{filepath}",
            f"b/{filepath}",
        )
    )
    return {
        "filepath": filepath,
        "replacements": count,
        "diff": diff[: RUN.limits.diff_chars],
        "diff_truncated": len(diff) > RUN.limits.diff_chars,
    }


@tool
async def get_status() -> dict:
    """Report the repository, baseline commit, current HEAD, changed paths, tool-call count and last saved patch."""
    _, changed = changes()
    return {
        "repo": str(RUN.repo),
        "baseline_commit": RUN.state["baseline_commit"],
        "head_commit": git(RUN.repo, "rev-parse", "HEAD").decode().strip(),
        "elapsed_s": round(time.time() - RUN.state["started_at"], 1),
        "tool_calls": RUN.state["tool_calls"],
        "command_timeout_s": RUN.limits.command_timeout_s,
        "changed_paths": changed,
        "last_patch": RUN.state["last_patch"],
    }


@tool
async def submit_patch() -> dict:
    """Save the diff from the baseline commit to the working tree, including new files, as patch.diff. Does not commit or stage anything."""
    patch, changed = changes()
    patch_path = RUN.run_dir / "patch.diff"
    atomic_write(patch_path, patch)
    RUN.state["last_patch"] = {
        "path": str(patch_path),
        "bytes": len(patch),
        "captured_at": time.time(),
    }
    return {
        "patch_path": str(patch_path),
        "bytes": len(patch),
        "changed_paths": changed,
        "empty": not patch,
    }


def on_sigterm(signum: int, frame: object) -> None:
    if RUN.pgid is not None:
        killpg(RUN.pgid, signal.SIGKILL)
    os._exit(0)


if __name__ == "__main__":
    RUN = open_run(sys.argv[1:])
    signal.signal(signal.SIGTERM, on_sigterm)
    mcp.run()
