# ollama-harness

A small coding agent for local Ollama models: fifteen file/search/shell/git tools, a
streaming terminal UI, and an OpenAI-compatible HTTP API.

```
uv tool install --force .          # from this directory: the dot matters
agent                              # REPL in the current directory
agent "add type hints to harness.py"
agent serve --port 8000            # OpenAI-compatible API
```

The distribution is `ollama-harness`; the command it installs is `agent`. Do not run
`uv tool install agent` — that resolves to an unrelated package of that name on PyPI.
For development, `uv tool install --force --editable .` picks up edits without
reinstalling.

## Layout

| file | contains |
| --- | --- |
| `tools.py` | the fifteen workspace tools — knows nothing about models or the console |
| `harness.py` | the `Harness` agent loop and its console rendering |
| `main.py` | all command-line code: `chat` and `serve` |
| `api.py` | FastAPI app implementing the OpenAI chat-completions contract |

`Harness.run_events()` is the loop as a stream of events (`step`, `token`, `usage`,
`message`, `tool`, `result`, `done`, `error`). `Harness.run()` renders those to the
terminal; the API serializes the same events as SSE. One loop, two front ends.

## Entering a task

A task is one line by default. For anything longer:

- `"""` on its own line opens a multi-line block; paste freely, then close it with
  another `"""`. Pasted lines are consumed by the block instead of each firing as its
  own task.
- `/file <path>` runs a task stored in a file — the practical way to re-run a prompt.
- `agent --file prompt.md` does the same from the shell, then exits.

## When a task stops

There is no step budget by default — a refactor that needs eighty tool calls gets
eighty. Runs end when the model answers, or when work stops making progress:

- `--retry-limit N` (default 5) — N consecutive *tool failures* abandons the task.
  Any successful call resets the counter.
- The same call failing four times with identical arguments abandons the task
  immediately, without waiting for the retry limit.
- `--max-steps N` is an optional hard cap; `0`, the default, means no limit.
- A 1,000-step ceiling exists as a backstop against a loop that never terminates.

Failing *tests* are not tool failures — `run_tests` returning `exit_code: 1` is a
successful call, so a test-fix-rerun loop can iterate as long as it needs to.

## Sub-agents

Off by default: `spawn_agent` is not in the tool list and the delegation rules are not
in the system prompt unless you pass `--subagents` or type `/subagents`. The model
cannot decide on its own to start delegating.

`spawn_agent(task, files)` runs one fresh agent to completion, sequentially, and
returns its report. A sub-agent starts with an empty conversation, cannot spawn agents
of its own, and its tokens are added to the parent's totals.

Ownership is enforced, not requested:

- Every sub-agent declares the files it owns. Claiming a file another sub-agent already
  owns is refused.
- A sub-agent that tries to write, patch or restore a file outside its own list gets an
  error telling it to report the needed change instead.

Sequential by design — one local Ollama instance gains nothing from concurrent
requests, and serial execution keeps the file-ownership guarantee simple.

## Permissions

Six tools mutate the workspace or the machine: `write_file`, `apply_patch`,
`run_command`, `run_tests`, `git_restore`, `install_dependency`.

- `--mode ask` (chat default) — confirm each call
- `--mode deny` (serve default) — read-only; refusals go back to the model, which
  turns the agent into a reviewer rather than a dead end
- `--mode allow` — no prompting

In the REPL, `/allow`, `/ask` and `/deny` switch mode mid-session.

## API

Server-side agent: one request runs the whole task. Tools execute in the server's
workspace, so the client needs no filesystem access and no loop of its own.

```
GET  /health
GET  /v1/models
POST /v1/chat/completions        # stream: true | false
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "what does harness.py do?"}]}'
```

Streaming responses are `chat.completion.chunk` SSE frames terminated by
`data: [DONE]`. The model's tokens and the agent's tool activity arrive as
`delta.reasoning_content`; only the final answer arrives as `delta.content`. Clients
that render a thinking pane (Open WebUI, LibreChat) show the agent working; clients
that ignore `reasoning_content` see just the answer.

`usage` reports Ollama's own `prompt_eval_count` / `eval_count`, summed across every
step of the agent loop — so a single request that took four model calls reports the
cost of all four. The same totals ride on the final SSE chunk.

### Serving safely

`serve` binds to `127.0.0.1` and defaults to `--mode deny`. `--mode allow` on a
reachable interface is remote code execution by design: pair it with `--api-key`
(or `HARNESS_API_KEY`) and keep the bind address local.

```bash
ollama-harness serve --mode allow --api-key "$(openssl rand -hex 16)" --workspace ~/code/project
```

## Notes

- The REPL prompt shows how full the context window is (`ask 34%>`); `/stats` prints
  context, steps, tokens in/out and model time for the session.
- The default model is `qwen3.8-coder:latest` for the ollama API.
- The model must support tool calling (`qwen3-coder`, `qwen2.5-coder`, `llama3.1+`,
  `mistral-nemo`). If tool calls only appear on the non-streaming path, use
  `--no-stream`.
- `--num-ctx` defaults to 32768. Ollama's own default is 2048, which silently
  truncates the tool schemas and looks like the model ignoring its tools.
- `OLLAMA_HOST` is often a bind address (`0.0.0.0:11434`); it is rewritten to a
  dialable one automatically.
