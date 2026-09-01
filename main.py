#!/usr/bin/env python3
"""Command line for the harness: an interactive coding agent, or an API server.

    harness                        # REPL in the current directory
    harness "add type hints"       # one task, then exit
    harness chat --provider openai # use OpenAI (or any compatible endpoint)
    harness serve --port 8000      # OpenAI-compatible API

Configuration comes from flags, then .env, then provider defaults. Providers:
ollama (default), openai (and OpenAI-compatible servers), openrouter.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.panel import Panel
from rich.table import Table

from harness import DEFAULT_NUM_CTX, RETRY_LIMIT, MODES, Harness, console, normalize_base_url
from providers import (DEFAULT_BASE_URLS, DEFAULT_MODELS, PROVIDERS,
                       api_key_from_env)

VERSION = "0.2.0"
COMMANDS = ("chat", "serve")
FENCE = '"""'


def pick_env(*names: str) -> str | None:
    """First env var among names that is set and non-empty, else None."""
    for name in names:
        if os.environ.get(name):
            return os.environ[name]
    return None


def provider_config(args: argparse.Namespace) -> tuple[str, str, str, str | None]:
    """Resolve provider, model, base url and api key: flags -> env -> defaults.

    HARNESS_* names are provider-agnostic; the legacy OLLAMA_MODEL / OLLAMA_HOST
    still work for the ollama provider.
    """
    provider = args.provider or pick_env("HARNESS_PROVIDER") or "ollama"
    model = (args.model or pick_env("HARNESS_MODEL")
             or (pick_env("OLLAMA_MODEL") if provider == "ollama" else None)
             or DEFAULT_MODELS[provider])
    base = (args.base_url
            or pick_env("HARNESS_BASE_URL")
            or (pick_env("OLLAMA_HOST") if provider == "ollama" else None)
            or DEFAULT_BASE_URLS[provider])
    key = args.api_key or api_key_from_env(provider)
    return provider, model, base, key


def read_block(first_line: str) -> str:
    """Collect a \"\"\"-fenced task. Pasting a block after the opening fence works:
    every pasted line is consumed here instead of firing as its own task."""
    body = [first_line[len(FENCE):]]
    console.print(f"[dim]multi-line — close with {FENCE} on its own line[/dim]")
    while True:
        try:
            line = console.input("[dim]··· [/dim]")
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
    parser.add_argument("--provider", choices=PROVIDERS, default=None,
                        help="model service: " + ", ".join(PROVIDERS)
                             + " (default: env HARNESS_PROVIDER or ollama)")
    parser.add_argument("--model", default=None,
                        help=f"model id (default per provider, e.g. {DEFAULT_MODELS['ollama']})")
    parser.add_argument("--base-url", default=None,
                        help="API endpoint (default per provider, e.g. "
                             + DEFAULT_BASE_URLS["ollama"] + ")")
    parser.add_argument("--api-key", default=None,
                        help="API key: model key for openai/openrouter; on serve "
                             "it is also the bearer key for /v1 routes "
                             "(default: HARNESS_API_KEY or the provider's env var)")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX,
                        help="context window (ollama only; default %(default)s)")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="hard cap on model calls per task (0 = no limit)")
    parser.add_argument("--retry-limit", type=int, default=RETRY_LIMIT,
                        help="consecutive tool failures before a task is abandoned")
    parser.add_argument("--temperature", type=float, default=0.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=Path(sys.argv[0]).stem or "harness",
        description="A small coding agent with pluggable model providers "
                    "(ollama, openai, openrouter).",
    )
    parser.add_argument("--version", action="version", version=f"ollama-harness {VERSION}")
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
    serve.add_argument("--cors-origin", action="append", dest="cors_origins",
                       help="allowed browser origin (repeatable; default: any)")
    return parser


def resolve_workspace(raw: str) -> Path:
    workspace = Path(raw).resolve()
    if not workspace.is_dir():
        console.print(f"[red]workspace not found: {workspace}[/red]")
        sys.exit(1)
    return workspace


def install_hint(provider: str) -> str:
    return ("uv pip install langchain-openai" if provider != "ollama"
            else "uv pip install langchain-ollama")


def print_tools(agent: Harness) -> None:
    table = Table(title=f"tools ({len(agent.tools)})", show_lines=False)
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


def make_agent(args: argparse.Namespace, workspace: Path,
               provider: str, model: str, base_url: str, api_key: str | None,
               stream: bool = True) -> Harness:
    try:
        return Harness(
            model=model, workspace=workspace, base_url=base_url,
            max_steps=args.max_steps, retry_limit=args.retry_limit,
            mode=args.mode, temperature=args.temperature,
            num_ctx=args.num_ctx, provider=provider, api_key=api_key,
            show_results=args.show_results, stream=stream,
            subagents=args.subagents,
        )
    except ImportError:
        console.print(f"[red]missing dependency:[/red] {install_hint(provider)}")
        sys.exit(1)


def startup_panel(provider: str, model: str, base_url: str, workspace: Path,
                  mode: str, num_ctx: int, tools: int, subagents: bool,
                  key_set: bool, stream: bool = True) -> None:
    console.print(Panel.fit(
        f"[bold]{provider}[/bold] · model [bold]{model}[/bold] @ {base_url}\n"
        f"context [bold]{num_ctx:,}[/bold] · workspace [bold]{workspace}[/bold]\n"
        f"mode [bold]{mode}[/bold] · {tools} tools"
        f"{'  ·  sub-agents [bold]on[/bold]' if subagents else ''}"
        f"{'  ·  [yellow]no api key[/yellow]' if provider != 'ollama' and not key_set else ''}",
        title="ollama-harness", border_style="green",
    ))


def run_chat(args: argparse.Namespace) -> None:
    workspace = resolve_workspace(args.workspace)
    provider, model, base_url, api_key = provider_config(args)
    base_url = normalize_base_url(base_url)
    stream = not args.no_stream
    agent = make_agent(args, workspace, provider, model, base_url, api_key,
                       stream=stream)

    startup_panel(provider, model, base_url, workspace, args.mode,
                  args.num_ctx, len(agent.tools), agent.subagents,
                  api_key is not None, stream)
    print_tools(agent)

    if args.file:
        task = read_prompt_file(args.file)
        if task:
            agent.run(task)
        return
    if args.query:
        agent.run(" ".join(args.query))
        return

    console.print('[dim]commands · ' + FENCE + ' opens a multi-line block · '
                  '/file <path> · /allow /ask /deny · /subagents · '
                  '/stats · /reset · /help · /quit[/dim]')
    while True:
        try:
            fill = (agent.context_used / agent.num_ctx * 100) if agent.num_ctx else 0
            gauge = f"[dim] {fill:.0f}%[/dim]" if agent.context_used else ""
            line = console.input(
                f"\n[bold magenta]{agent.mode}[/bold magenta]{gauge}[bold magenta]> [/bold magenta]").strip()
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
            if cmd == "help":
                console.print(
                    '[dim]  ' + FENCE + '  start a multi-line block\n'
                    '  /file <path>  run a task from a file\n'
                    '  /allow /ask /deny  switch permission mode\n'
                    '  /subagents  toggle the spawn_agent tool\n'
                    '  /stats  session context, steps, tokens\n'
                    '  /reset  clear conversation history\n'
                    '  /quit (or q)  exit[/dim]')
                continue
            if cmd == "reset":
                agent.messages = agent.messages[:1]
                agent.context_used = 0
                console.print("[dim]history cleared[/dim]")
                continue
            if line.startswith("/"):
                console.print(f"[yellow]unknown command:[/yellow] {line} — type /help")
                continue
        if line:
            agent.run(line)


def run_serve(args: argparse.Namespace) -> None:
    workspace = resolve_workspace(args.workspace)
    provider, model, base_url, _ = provider_config(args)
    api_key = args.api_key or pick_env("HARNESS_API_KEY")  # serve auth, not model auth
    base_url = normalize_base_url(base_url)
    try:
        import uvicorn

        from api import ServerConfig, create_app
    except ImportError:
        console.print("[red]missing dependency:[/red] uv pip install fastapi uvicorn")
        sys.exit(1)

    app = create_app(ServerConfig(
        model=model, base_url=base_url, workspace=workspace, mode=args.mode,
        num_ctx=args.num_ctx, max_steps=args.max_steps, retry_limit=args.retry_limit,
        provider=provider, api_key=api_key,
        cors_origins=args.cors_origins or ["*"],
    ))

    exposure = "localhost only" if args.host in ("127.0.0.1", "localhost") else "[bold red]all interfaces[/bold red]"
    console.print(Panel.fit(
        f"[bold]http://{args.host}:{args.port}/v1[/bold]  ({exposure})\n"
        f"[bold]{provider}[/bold] · model [bold]{model}[/bold] @ {base_url}\n"
        f"workspace [bold]{workspace}[/bold]\n"
        f"mode [bold]{args.mode}[/bold]"
        f"{'' if args.mode == 'deny' else '  [yellow](the agent can write files and run commands)[/yellow]'}\n"
        f"auth [bold]{'bearer key' if api_key else 'none'}[/bold]",
        title="ollama-harness serve", border_style="green",
    ))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def main(argv: list[str] | None = None) -> None:
    # .env in the launch directory wins; the project's own .env provides defaults.
    load_dotenv()
    load_dotenv(Path(__file__).resolve().parent / ".env")
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS and argv[0] not in ("-h", "--help", "--version"):
        argv.insert(0, "chat")   # bare invocation, flags, or a query all mean chat
    args = build_parser().parse_args(argv)
    (run_serve if args.command == "serve" else run_chat)(args)


if __name__ == "__main__":
    main()