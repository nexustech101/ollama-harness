#!/usr/bin/env python3
"""Command line for the harness: an interactive coding agent, or an API server.

    agent                       # REPL in the current directory
    agent "add type hints"      # one task, then exit
    agent serve --port 8000     # OpenAI-compatible API
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from rich.panel import Panel
from rich.table import Table

from harness import RETRY_LIMIT, MODES, Harness, console, normalize_base_url

COMMANDS = ("chat", "serve")
FENCE = '"""'


def read_block(first_line: str) -> str:
    """Collect a \"\"\"-fenced task. Pasting a block after the opening fence works:
    every pasted line is consumed here instead of firing as its own task."""
    body = [first_line[len(FENCE):]]
    console.print(f"[dim]multi-line — close with {FENCE} on its own line[/dim]")
    while True:
        try:
            line = console.input("")
        except (EOFError, KeyboardInterrupt):
            break
        if line.rstrip().endswith(FENCE):
            body.append(line.rstrip()[:-len(FENCE)])
            break
        body.append(line)
    return "\n".join(body).strip()


def read_prompt_file(raw: str) -> str:
    """Load a task from a file — handy for prompts you run more than once."""
    path = Path(raw.strip().strip('"').strip("'")).expanduser()
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError as e:
        console.print(f"[red]cannot read {path}: {e}[/red]")
        return ""
    console.print(f"[dim]{path.name}: {len(text.splitlines())} lines, {len(text):,} chars[/dim]")
    return text


def common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "qwen3.8-coder:latest"))
    parser.add_argument("--base-url", default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--num-ctx", type=int, default=128_000)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="hard cap on model calls per task (0 = no limit)")
    parser.add_argument("--retry-limit", type=int, default=RETRY_LIMIT,
                        help="consecutive tool failures before a task is abandoned")
    parser.add_argument("--temperature", type=float, default=0.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent",
        description="A small coding agent for local Ollama models.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    chat = sub.add_parser("chat", help="interactive coding agent (default)")
    common_args(chat)
    chat.add_argument("--mode", choices=MODES, default="ask",
                      help="allow: run mutating tools; ask: confirm each; deny: read-only")
    chat.add_argument("--no-stream", action="store_true", help="disable token streaming")
    chat.add_argument("--show-results", action="store_true",
                      help="print full tool output instead of a one-line summary")
    chat.add_argument("--subagents", action="store_true",
                      help="offer the spawn_agent tool (off by default)")
    chat.add_argument("--file", metavar="PATH",
                      help="run a task read from a file, then exit")
    chat.add_argument("query", nargs="*", help="run one task and exit")

    serve = sub.add_parser("serve", help="serve the OpenAI-compatible API")
    common_args(serve)
    serve.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost only)")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--mode", choices=("allow", "deny"), default="deny",
                       help="deny: read-only tools; allow: let the agent write and run commands")
    serve.add_argument("--api-key", default=os.getenv("HARNESS_API_KEY"),
                       help="require 'Authorization: Bearer <key>' on /v1 routes")
    serve.add_argument("--cors-origin", action="append", dest="cors_origins",
                       help="allowed browser origin (repeatable; default: any)")
    return parser


def resolve_workspace(raw: str) -> Path:
    workspace = Path(raw).resolve()
    if not workspace.is_dir():
        console.print(f"[red]workspace not found: {workspace}[/red]")
        sys.exit(1)
    return workspace


def print_tools(agent: Harness) -> None:
    table = Table(title="tools", show_lines=False)
    table.add_column("name", style="yellow")
    table.add_column("description", style="dim")
    for tool in agent.tools:
        table.add_row(tool.name, tool.description)
    console.print(table)


def print_stats(agent: Harness) -> None:
    used, limit = agent.context_used, agent.num_ctx
    pct = (used / limit * 100) if limit else 0.0
    totals = agent.stats
    table = Table(box=None, show_header=False, pad_edge=False)
    table.add_column(style="dim")
    table.add_column(justify="right")
    table.add_row("context", f"{used:,} / {limit:,}  ({pct:.0f}%)")
    table.add_row("messages", f"{len(agent.messages)}")
    table.add_row("steps", f"{totals['steps']:,}")
    table.add_row("tokens in", f"{totals['input']:,}")
    table.add_row("tokens out", f"{totals['output']:,}")
    table.add_row("model time", f"{totals['seconds']:.1f}s")
    console.print(table)


def run_chat(args: argparse.Namespace) -> None:
    workspace = resolve_workspace(args.workspace)
    base_url = normalize_base_url(args.base_url)
    try:
        agent = Harness(
            model=args.model, workspace=workspace, base_url=base_url,
            max_steps=args.max_steps, retry_limit=args.retry_limit,
            mode=args.mode, temperature=args.temperature,
            num_ctx=args.num_ctx, show_results=args.show_results, stream=not args.no_stream,
            subagents=args.subagents,
        )
    except ImportError:
        console.print("[red]missing dependency:[/red] uv pip install langchain-ollama")
        sys.exit(1)

    console.print(Panel.fit(
        f"model [bold]{args.model}[/bold] @ {base_url}\n"
        f"workspace [bold]{workspace}[/bold]\n"
        f"mode [bold]{agent.mode}[/bold]"
        f"{'  ·  sub-agents [bold]on[/bold]' if agent.subagents else ''}",
        title="agent harness", border_style="green",
    ))
    print_tools(agent)

    if args.file:
        task = read_prompt_file(args.file)
        if task:
            agent.run(task)
        return
    if args.query:
        agent.run(" ".join(args.query))
        return

    console.print(f'[dim]tasks · {FENCE} opens a multi-line block · /file <path>'
                  ' · /allow /ask /deny · /subagents · /stats · /reset · /quit[/dim]')
    while True:
        try:
            fill = (agent.context_used / agent.num_ctx * 100) if agent.num_ctx else 0
            gauge = f"[dim] {fill:.0f}%[/dim]" if agent.context_used else ""
            line = console.input(
                f"\n[bold magenta]{agent.mode}[/bold magenta]{gauge}[bold magenta]>[/bold magenta] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue

        if line.startswith(FENCE):
            line = read_block(line)
        elif line.lstrip("/").lower().startswith("file "):
            line = read_prompt_file(line.split(None, 1)[1])
        else:
            cmd = line.lstrip("/").lower()
            if cmd in ("quit", "exit", "q"):
                break
            if cmd in MODES:
                agent.mode = cmd
                console.print(f"[dim]mode: {cmd}[/dim]")
                continue
            if cmd == "subagents":
                agent.set_subagents(not agent.subagents)
                console.print(f"[dim]sub-agents: {'on' if agent.subagents else 'off'}[/dim]")
                continue
            if cmd == "stats":
                print_stats(agent)
                continue
            if cmd == "reset":
                agent.messages = agent.messages[:1]
                agent.context_used = 0
                console.print("[dim]history cleared[/dim]")
                continue
        if line:
            agent.run(line)


def run_serve(args: argparse.Namespace) -> None:
    workspace = resolve_workspace(args.workspace)
    base_url = normalize_base_url(args.base_url)
    try:
        import uvicorn

        from api import ServerConfig, create_app
    except ImportError:
        console.print("[red]missing dependency:[/red] uv pip install fastapi uvicorn")
        sys.exit(1)

    app = create_app(ServerConfig(
        model=args.model, base_url=base_url, workspace=workspace, mode=args.mode,
        num_ctx=args.num_ctx, max_steps=args.max_steps, retry_limit=args.retry_limit,
        api_key=args.api_key,
        cors_origins=args.cors_origins or ["*"],
    ))

    exposure = "localhost only" if args.host in ("127.0.0.1", "localhost") else "[bold red]all interfaces[/bold red]"
    console.print(Panel.fit(
        f"[bold]http://{args.host}:{args.port}/v1[/bold]  ({exposure})\n"
        f"model [bold]{args.model}[/bold] @ {base_url}\n"
        f"workspace [bold]{workspace}[/bold]\n"
        f"mode [bold]{args.mode}[/bold]"
        f"{'' if args.mode == 'deny' else '  [yellow](the agent can write files and run commands)[/yellow]'}\n"
        f"auth [bold]{'bearer key' if args.api_key else 'none'}[/bold]",
        title="ollama-harness serve", border_style="green",
    ))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS and argv[0] not in ("-h", "--help"):
        argv.insert(0, "chat")   # bare invocation, flags, or a query all mean chat
    args = build_parser().parse_args(argv)
    (run_serve if args.command == "serve" else run_chat)(args)


if __name__ == "__main__":
    main()
