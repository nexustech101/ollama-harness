#!/usr/bin/env python3
"""Command line for the harness: an interactive coding agent, or an API server.

    harness                        # REPL in the current directory (ollama)
    harness "add type hints"       # one task, then exit
    harness chat --provider openai # use OpenAI (or any compatible endpoint)
    harness serve --provider openrouter --config providers.yaml
    harness serve --port 8000      # OpenAI-compatible API

The default provider is ollama, configured from .env. Any other provider is
selected explicitly with --provider, which may name a built-in (ollama, openai,
openrouter) or an entry from providers.yaml; the config file is read only then.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.panel import Panel
from rich.table import Table

from harness import RETRY_LIMIT, MODES, Harness, console, normalize_base_url
from providers import (DEFAULT_BASE_URLS, DEFAULT_MODELS, DEFAULT_NUM_CTX,
                       PROVIDERS, Provider, find_config, load_config,
                       resolve_provider)

VERSION = "0.2.0"
COMMANDS = ("chat", "serve", "init-provider")
FENCE = '"""'


def pick_env(*names: str) -> str | None:
    """First env var among names that is set and non-empty, else None."""
    for name in names:
        if os.environ.get(name):
            return os.environ[name]
    return None


def load_provider_config(args: argparse.Namespace) -> tuple[dict, Path | None]:
    """Locate and parse the providers YAML; returns (config, path) or ({}, None)."""
    cfg_path = find_config(args.config)
    if args.config and cfg_path is None:
        console.print(f"[red]provider config not found: {args.config}[/red]")
        sys.exit(1)
    if cfg_path is None:
        return {}, None
    try:
        return load_config(cfg_path), cfg_path
    except (ValueError, OSError) as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)


def _ask(prompt: str, default: str | None = None) -> str:
    """Prompt for a value with an optional default (empty input takes default)."""
    suffix = f" [{default}]" if default else ""
    try:
        value = console.input(f"{prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        return default or ""
    return value or default or ""


def _ask_secret(prompt: str) -> str | None:
    """Prompt for a secret; blank means 'leave it out' (env vars will be used)."""
    try:
        value = console.input(f"{prompt} (blank to skip): ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    return value or None


def add_provider_to_config(path: Path, name: str, opts: dict) -> None:
    """Merge one provider entry into the YAML at path (creates the file if needed).

    Keeps existing entries and comments; writes via yaml.dump so the result is
    always valid. The file is written on a new line-block, opening a mapping.
    """
    import yaml

    from providers import load_config

    existing: dict = {"providers": {}}
    if path.is_file():
        try:
            existing = load_config(path)
        except ValueError:
            pass  # start fresh if the file is unparsable; do not clobber below
    existing.setdefault("providers", {})
    existing["providers"][name] = opts

    # re-dump, preserving key order (existing entries first)
    block = yaml.dump(existing, sort_keys=False, allow_unicode=True).rstrip()
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# Named providers for the harness. Read only when --provider NAME is given.\n")
        fh.write("# Without --provider the harness always uses ollama from .env.\n")
        fh.write(block + "\n")


def run_init_provider(args: argparse.Namespace) -> None:
    """Add a provider to providers.yaml — interactively, or fully via flags.

    ``harness init-provider`` asks for each value; ``harness init-provider
    --name openrouter --base-url https://openrouter.ai/api/v1 --model
    openai/gpt-5.5 [--api-key sk-...]`` writes it without prompting.
    """
    from providers import PROVIDERS, build_provider_entry

    console.print("[bold]Initialize a provider in providers.yaml[/bold]")
    path = (Path(args.config).expanduser() if args.config
            else find_config(None) or Path.cwd() / "providers.yaml")

    # Fully flagged runs (--name + --base-url) skip every prompt; otherwise ask.
    full = bool(args.name and args.base_url)
    name = args.name or _ask("provider name (alias)", "openrouter")
    base = args.base_url or _ask("base_url (API endpoint)",
                                 "https://openrouter.ai/api/v1")
    model = args.model or (None if full else
                           _ask("default model (override per run with --model)"))
    # Built-in names inherit their protocol; custom names default to openai.
    if name in PROVIDERS:
        kind: str | None = None
    else:
        kind = args.kind or ("openai" if full else _ask(
            f"wire protocol ({', '.join(PROVIDERS)}; custom names default to openai)",
            "openai"))
        if kind not in PROVIDERS:
            console.print(f"[red]unknown kind {kind!r} — must be one of {', '.join(PROVIDERS)}[/red]")
            sys.exit(1)
    api_key = args.api_key
    if api_key is None and not full:
        try:
            store = console.input("store an api_key in this file? [y/N]: ").strip().lower() in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            store = False
        api_key = _ask_secret("api_key (sk-…)") if store else None

    entry = {
        k: v for k, v in build_provider_entry(
            kind=kind, base_url=base, api_key=api_key, model=model).items()
        if v is not None
    }
    try:
        resolve_provider(name, config={"providers": {name: entry}})
    except ValueError as e:
        console.print(f"[red]invalid entry: {e}[/red]")
        sys.exit(1)
    add_provider_to_config(path, name, entry)
    console.print(f"[green]wrote {name!r} to {path}[/green]")
    console.print(f"use it with: [bold]harness --provider {name}[/bold] (or --model to override)")


def provider_config(args: argparse.Namespace) -> tuple[Provider, Path | None]:
    """Resolve one concrete Provider from flags + env.

    The providers.yaml config is read only when --provider is given (a named
    entry or a built-in whose settings the config may override). Without
    --provider the result is ollama, configured from .env.
    """
    if args.provider:
        config, cfg_path = load_provider_config(args)
    else:
        if args.config:
            console.print("[yellow]--config needs --provider NAME; ignoring "
                          "--config (default provider is ollama)[/yellow]")
        config, cfg_path = {}, None
    try:
        prov = resolve_provider(
            args.provider, model=args.model, base_url=args.base_url,
            api_key=args.api_key, num_ctx=args.num_ctx,
            temperature=args.temperature, config=config,
        )
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    return prov, cfg_path


def provider_label(prov: Provider) -> str:
    """Display name: the alias, with its wire protocol when they differ."""
    if prov.label and prov.label != prov.name:
        return f"{prov.label} ({prov.name})"
    return prov.label or prov.name


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
    parser.add_argument("--provider", default=None,
                        help="provider: ollama is the default; pick openai, "
                             "openrouter, or a name from providers.yaml")
    parser.add_argument("--config", metavar="PATH", default=None,
                        help="providers.yaml (default: $HARNESS_CONFIG, ./providers.yaml, "
                             "~/.config/harness/providers.yaml, or the platform app-data "
                             "dir under harness/providers.yaml)")
    parser.add_argument("--model", default=None,
                        help=f"model id (overrides the config default, e.g. {DEFAULT_MODELS['ollama']})")
    parser.add_argument("--base-url", default=None,
                        help="API endpoint (overrides the config default, e.g. "
                             + DEFAULT_BASE_URLS["ollama"] + ")")
    parser.add_argument("--api-key", default=None,
                        help="API key: overrides the config/env key; on serve it is "
                             "also the bearer key for /v1 routes")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--num-ctx", type=int, default=None,
                        help="context window for ollama (default: the provider's "
                             f"config num_ctx or {DEFAULT_NUM_CTX:,})")
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

    initp = sub.add_parser("init-provider",
                           help="interactively add a provider to providers.yaml")
    initp.add_argument("--config", metavar="PATH", default=None,
                       help="target file (default: ./providers.yaml or the default locations)")
    initp.add_argument("--name", default=None, help="provider alias (default: prompted)")
    initp.add_argument("--base-url", default=None, help="API endpoint (default: prompted)")
    initp.add_argument("--model", default=None, help="default model (default: prompted)")
    initp.add_argument("--kind", default=None, help="wire protocol (default: openai)")
    initp.add_argument("--api-key", default=None, help="store this key in the file")
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


def make_agent(args: argparse.Namespace, workspace: Path, prov: Provider,
               stream: bool = True) -> Harness:
    try:
        return Harness(
            model=prov.model, workspace=workspace, base_url=prov.endpoint,
            max_steps=args.max_steps, retry_limit=args.retry_limit,
            mode=args.mode, temperature=prov.temperature,
            num_ctx=prov.num_ctx, provider=prov.name, api_key=prov.api_key,
            show_results=args.show_results, stream=stream,
            subagents=args.subagents,
        )
    except ImportError:
        console.print(f"[red]missing dependency:[/red] {install_hint(prov.name)}")
        sys.exit(1)


def startup_panel(prov: Provider, base_url: str, workspace: Path,
                  mode: str, tools: int, subagents: bool,
                  key_set: bool, cfg_path: Path | None,
                  stream: bool = True) -> None:
    console.print(Panel.fit(
        f"[bold]{provider_label(prov)}[/bold] · model [bold]{prov.model}[/bold] @ {base_url}\n"
        f"context [bold]{prov.num_ctx:,}[/bold] · workspace [bold]{workspace}[/bold]\n"
        f"mode [bold]{mode}[/bold] · {tools} tools"
        f"{'  ·  sub-agents [bold]on[/bold]' if subagents else ''}"
        f"{'  ·  config [dim]' + str(cfg_path) + '[/dim]' if cfg_path else ''}"
        f"{'  ·  [yellow]no api key[/yellow]' if prov.name != 'ollama' and not key_set else ''}",
        title="ollama-harness", border_style="green",
    ))


def run_chat(args: argparse.Namespace) -> None:
    workspace = resolve_workspace(args.workspace)
    prov, cfg_path = provider_config(args)
    base_url = normalize_base_url(prov.endpoint)
    prov.base_url = base_url
    stream = not args.no_stream
    agent = make_agent(args, workspace, prov, stream=stream)

    startup_panel(prov, base_url, workspace, args.mode,
                  len(agent.tools), agent.subagents,
                  prov.api_key is not None, cfg_path, stream)
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
    prov, cfg_path = provider_config(args)
    base_url = normalize_base_url(prov.endpoint)
    prov.base_url = base_url
    auth_key = args.api_key or pick_env("HARNESS_API_KEY")  # bearer auth for /v1
    model_key = args.api_key or prov.api_key               # key the model call uses
    try:
        import uvicorn

        from api import ServerConfig, create_app
    except ImportError:
        console.print("[red]missing dependency:[/red] uv pip install fastapi uvicorn")
        sys.exit(1)

    app = create_app(ServerConfig(
        model=prov.model, base_url=base_url, workspace=workspace, mode=args.mode,
        num_ctx=prov.num_ctx, max_steps=args.max_steps, retry_limit=args.retry_limit,
        provider=prov.name, api_key=auth_key, model_api_key=model_key,
        cors_origins=args.cors_origins or ["*"],
    ))

    keyed = prov.api_key is not None or auth_key is not None
    exposure = "localhost only" if args.host in ("127.0.0.1", "localhost") else "[bold red]all interfaces[/bold red]"
    console.print(Panel.fit(
        f"[bold]http://{args.host}:{args.port}/v1[/bold]  ({exposure})\n"
        f"[bold]{provider_label(prov)}[/bold] · model [bold]{prov.model}[/bold] @ {base_url}\n"
        f"workspace [bold]{workspace}[/bold]\n"
        f"mode [bold]{args.mode}[/bold]"
        f"{'' if args.mode == 'deny' else '  [yellow](the agent can write files and run commands)[/yellow]'}\n"
        f"auth [bold]{'bearer key' if auth_key else 'none'}[/bold]"
        f"{'  ·  [yellow]no model key[/yellow]' if prov.name != 'ollama' and not keyed else ''}"
        f"{'  ·  config [dim]' + str(cfg_path) + '[/dim]' if cfg_path else ''}",
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
    if args.command == "init-provider":
        run_init_provider(args)
    else:
        (run_serve if args.command == "serve" else run_chat)(args)


if __name__ == "__main__":
    main()