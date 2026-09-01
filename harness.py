#!/usr/bin/env python3
"""The agent loop: one model call, its tool calls, repeat until it answers.

`Harness.run_events()` yields the loop as events; `Harness.run()` renders those
to the console. Tools live in tools.py; the CLI lives in main.py.
"""

from __future__ import annotations

import json
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Optional

from langchain_core.tools import BaseTool
from pydantic import Field, field_validator
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from providers import Provider
from tools import DESTRUCTIVE, ToolArgs, create_tools, describe_environment, truncate

console = Console()

TOOL_RESULT_LIMIT = 6_000  # cap on what re-enters the model's context
MODES = ("allow", "ask", "deny")   # run mutating tools / confirm each / refuse them
SPINNER = "|/-\\"
THINK_RE = re.compile(r"<think>.*?</think>", re.S)
THINK_TAGS = re.compile(r"</?think>")
REPEAT_LIMIT = 3      # identical failing tool calls tolerated before abandoning
RETRY_LIMIT = 5       # consecutive tool failures tolerated before abandoning
STEP_CEILING = 1_000  # backstop against a loop that never terminates
DEFAULT_NUM_CTX = 200_000  # single source of truth for the default context window


class StreamView:
    """One live line: step, elapsed, tokens, idle gap, and the tail of what the
    model is emitting. A frozen token count with a growing idle gap means stuck."""

    def __init__(self, step: int, max_steps: int):
        self.step, self.max_steps = step, max_steps
        self.text = ""
        self.tokens = 0
        self.start = self.last = time.monotonic()
        self.frame = 0

    def reset(self, step: int) -> None:
        self.step = step
        self.text = ""
        self.tokens = 0
        self.start = self.last = time.monotonic()

    def add(self, delta: str) -> None:
        self.text += delta
        self.tokens += 1
        self.last = time.monotonic()

    def __rich__(self) -> Text:
        now = time.monotonic()
        self.frame += 1
        idle = now - self.last
        where = f"step {self.step}" + (f"/{self.max_steps}" if self.max_steps else "")
        head = (f"{SPINNER[self.frame % len(SPINNER)]} {where}  "
                f"{now - self.start:5.1f}s  {self.tokens:>4} tok  ")
        tail = " ".join(self.text.split())[-400:] or "waiting for first token"
        line = Text(head, style="cyan")
        if idle > 3:
            line.append(f"[idle {idle:.0f}s] ", style="bold yellow")
        line.append(tail, style="dim")
        line.no_wrap, line.overflow = True, "ellipsis"
        return line


def fmt_args(args: dict, limit: int = 60) -> str:
    """Tool arguments on one line, with long values (file contents) elided."""
    parts = []
    for key, value in args.items():
        flat = " ".join(str(value).split())
        parts.append(f"{key}={flat[:limit]}…" if len(flat) > limit else f"{key}={flat}")
    return "  ".join(parts)


def summarize(result: str, limit: int = 120) -> str:
    """One line standing in for a tool result: its first line plus a size."""
    lines = result.splitlines() or [""]
    head = " ".join(lines[0].split())[:limit]
    if result.startswith("Error"):
        return " / ".join(" ".join(l.split()) for l in lines[:3] if l.strip())[:3 * limit]
    if len(lines) == 1:
        return head or f"{len(result):,} chars"
    return f"{head}  ·  {len(lines)} lines, {len(result):,} chars"


def text_of(message: Any) -> str:
    """Plain text from a message whose content is a string or a list of blocks."""
    content = getattr(message, "content", "") or ""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            parts.append(block.get("text") or block.get("thinking") or "")
    return "".join(parts)


def usage_from(message: Any) -> dict[str, Any]:
    """Real counts from Ollama: usage_metadata if present, else raw eval counts."""
    usage = getattr(message, "usage_metadata", None) or {}
    meta = getattr(message, "response_metadata", None) or {}
    return {
        "input": usage.get("input_tokens") or meta.get("prompt_eval_count") or 0,
        "output": usage.get("output_tokens") or meta.get("eval_count") or 0,
        "eval_seconds": (meta.get("eval_duration") or 0) / 1e9,
    }


def usage_line(event: dict) -> Text:
    """One dim line: how full the window is, how much came back, how fast."""
    used, limit = event["context"], event["limit"]
    pct = (used / limit * 100) if limit else 0.0
    seconds = event["eval_seconds"] or event["seconds"]
    speed = event["output"] / seconds if seconds > 0.05 else 0.0
    line = Text("  ctx ", style="dim")
    line.append(f"{used:,}/{limit:,} {pct:.0f}%",
                style="bold red" if pct >= 90 else "yellow" if pct >= 70 else "dim")
    rate = f"  ·  {speed:.0f} tok/s" if speed else ""
    line.append(f"  ·  {event['input']:,} in  {event['output']:,} out"
                f"{rate}  ·  {event['seconds']:.1f}s", style="dim")
    return line


def normalize_base_url(url: str) -> str:
    """OLLAMA_HOST is often a bind address ('0.0.0.0:11434'); make it dialable."""
    url = url.strip().rstrip("/")
    if "://" not in url:
        url = "http://" + url
    return url.replace("://0.0.0.0", "://127.0.0.1").replace("://[::]", "://127.0.0.1")


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a coding agent working in the directory {workspace}.

{environment}

Rules:
- Every shell command must be valid for the environment above. Do not use commands,
  paths or quoting from a different operating system.
- Use tools to inspect the workspace before you make claims about it. Never guess file contents.
- Prefer apply_patch for edits to existing files; use write_file for new files or full rewrites.
- All paths are relative to the workspace root.
- Read files in ranges: use search_files to locate the relevant lines, then read_file
  with start_line/end_line. Do not read large files whole.
- Take one step at a time: call a tool, read the result, then decide the next step.
- When the task is complete, reply with a short plain-text summary and no further tool calls.
"""


SUBAGENT_RULES = """

Sub-agents:
- spawn_agent runs one fresh agent to completion and returns its report. Use it only
  for work the user asked to be delegated.
- Each sub-agent must own every file it will change, exclusively. Two agents touching
  the same file, module or test file corrupt each other's work; overlapping ownership
  is refused, and re-delegating a file another sub-agent already owns will fail.
- Split by module, not by step: "write wordcount/core.py and its tests" is a good
  slice, "write the code while another writes its tests" is not.
- If the work cannot be split without sharing files, do it yourself.
- Sub-agents start fresh: they cannot see this conversation and cannot spawn agents of
  their own, so the task description must contain everything they need.
- Integration stays yours. After the sub-agents finish, verify the result as a whole
  (run the tests) before you answer.
"""


class SpawnAgentArgs(ToolArgs):
    task: str = Field(description="Everything the sub-agent needs to know, in full. "
                                  "It cannot see this conversation.")
    files: list[str] = Field(description="Files this sub-agent owns exclusively and is "
                                         "allowed to create or modify.")

    @field_validator("files", mode="before")
    @classmethod
    def _listify(cls, value: Any) -> Any:
        return [value] if isinstance(value, str) else value


class SpawnAgentTool(BaseTool):
    name: str = "spawn_agent"
    description: str = (
        "Delegate a self-contained slice of work to a fresh sub-agent that owns the "
        "listed files exclusively. Runs to completion and returns its report."
    )
    args_schema: type = SpawnAgentArgs
    parent: Any = None

    def _run(self, task: str, files: list[str]) -> str:
        return self.parent.delegate(task, files)


class Harness:
    def __init__(
            self,
            model: str,
            workspace: Path,
            base_url: str,
            *,
            max_steps: int = 0,          # 0 = no limit
            retry_limit: int = RETRY_LIMIT,
            mode: str = "ask",
            temperature: float = 0.0,
            num_ctx: int = DEFAULT_NUM_CTX,
            provider: str = "ollama",
            api_key: Optional[str] = None,
            show_results: bool = False,
            stream: bool = True,
            subagents: bool = False):
        self.model = model
        self.base_url = base_url
        self.provider = provider
        self.api_key = api_key
        self.temperature = temperature
        self.subagents = subagents
        self.claimed: dict[str, int] = {}   # file -> sub-agent that owns it
        self.owned: Optional[set[str]] = None   # set on a sub-agent: its writable files
        self.children = 0
        self.workspace = workspace
        self.max_steps = max_steps
        self.retry_limit = retry_limit
        self.mode = mode
        self.stream = stream
        self.live: Optional[Live] = None
        self.num_ctx = num_ctx
        self.context_used = 0        # size of the conversation as of the last call
        self.stats = {"steps": 0, "input": 0, "output": 0, "seconds": 0.0}
        self.repeats: dict[str, int] = {}
        self.failures = 0            # consecutive failing tool calls
        self.show_results = show_results
        self.bind_tools()
        self.messages: list[BaseMessage] = [SystemMessage(self.system_prompt())]

    def system_prompt(self) -> str:
        prompt = SYSTEM_PROMPT.format(
            workspace=self.workspace.resolve(),
            environment=describe_environment(self.workspace.resolve()),
        )
        return prompt + SUBAGENT_RULES if self.subagents else prompt

    def bind_tools(self) -> None:
        """(Re)build the tool set and the model. spawn_agent appears only when
        sub-agents are on. The model itself comes from the provider layer."""
        self.tools = create_tools(self.workspace)
        if self.subagents:
            self.tools.append(SpawnAgentTool(parent=self))
        self.tool_map = {t.name: t for t in self.tools}
        self.llm = Provider(
            name=self.provider, model=self.model, base_url=self.base_url,
            api_key=self.api_key, num_ctx=self.num_ctx,
            temperature=self.temperature,
        ).build().bind_tools(self.tools)

    def set_subagents(self, enabled: bool) -> None:
        self.subagents = enabled
        self.bind_tools()
        self.messages[0] = SystemMessage(self.system_prompt())

    def delegate(self, task: str, files: list[str]) -> str:
        """Run one sub-agent to completion over files it owns exclusively."""
        owned = sorted({f.replace("\\", "/").strip().lstrip("./") for f in files if f.strip()})
        if not owned:
            return "Error: list the files this sub-agent will own."
        clash = [f for f in owned if f in self.claimed]
        if clash:
            return (f"Error: {', '.join(clash)} already owned by sub-agent "
                    f"{self.claimed[clash[0]]}. Two agents must not share files — "
                    "re-split the work or handle this part yourself.")

        self.children += 1
        index = self.children
        child = Harness(
            model=self.model, workspace=self.workspace, base_url=self.base_url,
            max_steps=self.max_steps, retry_limit=self.retry_limit, mode=self.mode,
            temperature=self.temperature, num_ctx=self.num_ctx,
            provider=self.provider, api_key=self.api_key,
            show_results=self.show_results, stream=self.stream, subagents=False,
        )
        child.owned = set(owned)        # enforced, not merely requested
        child.live = self.live          # so approval prompts pause the parent's display
        child.messages[0] = SystemMessage(
            child.system_prompt()
            + f"\n\nYou are sub-agent {index}, working alone on one slice of a larger task.\n"
              f"You own these files and may create or modify only them: {', '.join(owned)}.\n"
              "Read anything you need, but change nothing else. Finish by reporting what "
              "you changed and whether you verified it."
        )

        out = self.live.console if self.live is not None else console
        out.print(Text(f"  ⟶ sub-agent {index} owns {', '.join(owned)}", style="magenta"))
        answer = ""
        for event in child.run_events(task):
            kind = event["type"]
            if kind == "tool":
                out.print(Text(f"    → {event['name']}  {fmt_args(event['args'])}", style="dim"))
            elif kind == "result":
                out.print(Text(f"      {event['summary']}", style="dim"))
            elif kind == "error":
                out.print(Text(f"    {event['text']}", style="red"))
            elif kind == "done":
                answer = event["text"]

        for key in ("steps", "input", "output", "seconds"):
            self.stats[key] += child.stats[key]   # the child's cost is the parent's cost
        for path in owned:
            self.claimed[path] = index
        out.print(Text(f"  ⟵ sub-agent {index} finished — {child.stats['output']:,} tokens, "
                       f"{child.stats['seconds']:.0f}s", style="magenta"))
        return truncate(answer or "(the sub-agent returned no report)", TOOL_RESULT_LIMIT)

    def approve(self, name: str, args: dict) -> Optional[str]:
        """None if the call may proceed, else the reason it may not."""
        if name not in DESTRUCTIVE or self.mode == "allow":
            return None
        if self.mode == "deny":
            return f"{name} is blocked: the session is read-only. Say what you would change instead."
        if not sys.stdin.isatty():
            return f"{name} needs confirmation but no terminal is attached. Re-run with --mode allow."
        if self.live is not None:
            self.live.stop()          # release the terminal for the prompt
        try:
            console.print(Text.assemble(("approve ", "yellow"), (name, "bold yellow"),
                                        ("  " + fmt_args(args, 200), "dim")))
            answer = console.input("[yellow]  run it? [y/N] [/yellow]").strip().lower()
        finally:
            if self.live is not None:
                self.live.start()
        return None if answer in ("y", "yes") else f"the user denied {name}. Try a different approach."

    @staticmethod
    def _norm(path: str) -> str:
        return str(path).replace("\\", "/").strip().lstrip("./")

    def write_targets(self, name: str, args: dict) -> list[str]:
        """Files a tool call would modify, for the sub-agent ownership check."""
        if name in ("write_file", "git_restore"):
            return [self._norm(args.get("path", ""))]
        if name == "apply_patch":
            return [self._norm(m) for m in re.findall(r"^\+\+\+ b/(.+)$",
                                                     str(args.get("patch", "")), flags=re.M)]
        return []

    def call_tool(self, name: str, args: dict) -> str:
        tool = self.tool_map.get(name)
        if tool is None:
            return f"Error: unknown tool {name!r}. Available: {', '.join(self.tool_map)}"
        if self.owned is not None:
            trespass = [t for t in self.write_targets(name, args) if t not in self.owned]
            if trespass:
                return (f"Error: you own only {', '.join(sorted(self.owned))}. "
                        f"{', '.join(trespass)} belongs to another agent — do not modify it; "
                        "say in your report what needs to change there.")
        denied = self.approve(name, args)
        if denied:
            return f"Error: {denied}"
        try:
            return str(tool.invoke(args))
        except Exception as e:  # tool errors go back to the model, not the traceback
            return f"Error in {name}: {type(e).__name__}: {e}"

    def trim(self, keep: int = 24) -> None:
        """Keep the system prompt plus the most recent messages, dropping any
        leading orphan tool result left by the cut."""
        if len(self.messages) <= keep + 1:
            return
        tail = self.messages[-keep:]
        while tail and isinstance(tail[0], ToolMessage):
            tail.pop(0)
        self.messages = self.messages[:1] + tail

    def _think(self, step: int) -> Iterator[dict]:
        """One model call: token events, then a single reply event."""
        started = time.monotonic()
        if not self.stream:
            reply = self.llm.invoke(self.messages)
            yield {"type": "reply", "message": reply, "tokens": 0,
                   "seconds": time.monotonic() - started}
            return

        merged = None
        tokens = 0
        for chunk in self.llm.stream(self.messages):
            merged = chunk if merged is None else merged + chunk
            piece = text_of(chunk)
            piece += (chunk.additional_kwargs or {}).get("reasoning_content") or ""
            if piece:
                tokens += 1
                yield {"type": "token", "text": piece}
        if merged is None:
            raise RuntimeError("the model returned an empty stream")
        yield {
            "type": "reply",
            "message": AIMessage(content=merged.content, tool_calls=merged.tool_calls,
                                 response_metadata=merged.response_metadata,
                                 usage_metadata=merged.usage_metadata,
                                 additional_kwargs=merged.additional_kwargs),
            "tokens": tokens,
            "seconds": time.monotonic() - started,
        }

    def run_events(self, query: str) -> Iterator[dict]:
        """The agent loop as a stream of events: step, token, usage, message,
        tool, result, done, error. run() renders these; the API serializes them."""
        self.trim()
        self.repeats.clear()
        self.failures = 0
        self.messages.append(HumanMessage(query))
        step = 0
        while True:
            step += 1
            if self.max_steps and step > self.max_steps:
                yield {"type": "error",
                       "text": f"stopped at the {self.max_steps}-step limit (--max-steps)"}
                return
            if step > STEP_CEILING:
                yield {"type": "error",
                       "text": f"stopped at the {STEP_CEILING:,}-step safety ceiling"}
                return
            yield {"type": "step", "step": step}
            reply = None
            try:
                for event in self._think(step):
                    if event["type"] == "reply":
                        reply = event["message"]
                        used = usage_from(reply)
                        self.context_used = used["input"] + used["output"]
                        self.stats["steps"] += 1
                        self.stats["input"] += used["input"]
                        self.stats["output"] += used["output"]
                        self.stats["seconds"] += event["seconds"]
                        yield {"type": "usage", **used, "seconds": event["seconds"],
                               "chunks": event["tokens"], "context": self.context_used,
                               "limit": self.num_ctx}
                    else:
                        yield event
            except KeyboardInterrupt:
                yield {"type": "error", "text": "interrupted"}
                return
            except Exception as e:
                text = f"LLM call failed: {type(e).__name__}: {e}"
                if self.provider != "ollama" and (
                        "401" in text or "authentication" in text.lower()):
                    text += (" — set HARNESS_API_KEY (or the provider's key env "
                             "var) and check --base-url.")
                yield {"type": "error", "text": text}
                return

            self.messages.append(reply)
            raw = text_of(reply)
            text = THINK_RE.sub("", raw).strip()
            thinking = False
            if not text and not reply.tool_calls:
                # the whole answer stayed inside <think>, or Ollama put it in the
                # separate reasoning channel: show it rather than nothing
                text = (THINK_TAGS.sub("", raw).strip()
                        or ((reply.additional_kwargs or {}).get("reasoning_content") or "").strip())
                thinking = bool(text)
            if text:
                yield {"type": "message", "text": text, "thinking": thinking}
            if not reply.tool_calls:
                if not text:
                    # tokens were generated but nothing usable came back: report what
                    # the model returned, then ask once more before giving up
                    self.failures += 1
                    meta = reply.response_metadata or {}
                    yield {"type": "error", "text":
                           f"empty turn (done_reason={meta.get('done_reason')}, "
                           f"content={type(reply.content).__name__}, "
                           f"{usage_from(reply)['output']} tokens) — asking again"}
                    if self.failures >= self.retry_limit:
                        yield {"type": "error", "text":
                               f"abandoned after {self.failures} unproductive turns"}
                        return
                    self.messages.append(HumanMessage(
                        "Your last turn contained no text and no tool call. Reply now "
                        "with your final answer as plain text, no tool calls."))
                    continue
                yield {"type": "done", "text": text}
                return

            for tc in reply.tool_calls:
                name, args = tc["name"], tc.get("args") or {}
                yield {"type": "tool", "name": name, "args": args}
                result = self.call_tool(name, args)
                # only failures count against the limits: repeatedly running the
                # same tests or re-checking git status is normal, healthy work
                signature = f"{name}:{json.dumps(args, sort_keys=True, default=str)}"
                if result.startswith("Error"):
                    seen = self.repeats[signature] = self.repeats.get(signature, 0) + 1
                    self.failures += 1
                    if seen > 1:
                        result += (f"\n\nThis exact call has now failed {seen} times. Do not "
                                   "repeat it: change the arguments, use a different tool, or "
                                   "explain why you cannot proceed.")
                else:
                    seen = 0
                    self.repeats.pop(signature, None)
                    self.failures = 0
                yield {"type": "result", "name": name, "result": result,
                       "summary": summarize(result)}
                if seen > REPEAT_LIMIT:
                    yield {"type": "error", "text":
                           f"abandoned: {name} failed {seen} times with identical arguments"}
                    return
                if self.failures >= self.retry_limit:
                    yield {"type": "error", "text":
                           f"abandoned: {self.failures} tool calls failed in a row "
                           f"(--retry-limit {self.retry_limit})"}
                    return
                self.messages.append(ToolMessage(
                    content=truncate(result, TOOL_RESULT_LIMIT),
                    tool_call_id=tc.get("id") or name,
                ))

    def run(self, query: str) -> str:
        """Render run_events() to the console; return the final answer."""
        view = StreamView(0, self.max_steps)
        final = ""
        with Live(view, console=console, refresh_per_second=8, transient=True) as live:
            self.live = live
            try:
                for ev in self.run_events(query):
                    kind = ev["type"]
                    if kind == "step":
                        view.reset(ev["step"])
                    elif kind == "token":
                        view.add(ev["text"])
                    elif kind == "usage":
                        live.console.print(usage_line(ev))
                    elif kind == "message":
                        live.console.print(Panel(
                            ev["text"],
                            title="assistant (thinking only)" if ev.get("thinking") else "assistant",
                            border_style="yellow" if ev.get("thinking") else "cyan"))
                    elif kind == "tool":
                        live.console.print(Text.assemble(
                            ("→ ", "yellow"), (ev["name"], "bold yellow"),
                            ("  " + fmt_args(ev["args"]), "dim")))
                    elif kind == "result":
                        if self.show_results:
                            live.console.print(Panel(truncate(ev["result"], 1500),
                                                     border_style="dim", title="result"))
                        else:
                            live.console.print(Text("  " + ev["summary"], style="dim"))
                    elif kind == "error":
                        live.console.print(f"[red]{ev['text']}[/red]")
                    elif kind == "done":
                        final = ev["text"]
                        if not final:
                            live.console.print(Text(
                                "  the model ended its turn with no text and no tool call",
                                style="yellow"))
            finally:
                self.live = None
        totals = self.stats
        console.print(Text(
            f"  session  {totals['steps']} steps  ·  {totals['input']:,} in"
            f"  {totals['output']:,} out  ·  {totals['seconds']:.0f}s", style="dim"))
        return final
