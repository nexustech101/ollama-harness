# ollama-harness

A small coding agent with pluggable model providers: file/search/shell/git
tools, a streaming terminal UI, and an OpenAI-compatible HTTP API. The same
agent loop runs against local Ollama models, OpenAI, OpenRouter, or any
OpenAI-compatible endpoint (vLLM, LM Studio, Together, Groq, …).

```
uv tool install --force .          # from this directory: the dot matters
harness                            # REPL in the current directory
harness "add type hints to harness.py"
harness serve --port 8000          # OpenAI-compatible API
```

The distribution is `ollama-harness`; the command it installs is `harness`.
For development, `uv tool install --force --editable .` picks up edits without
reinstalling. Do not run `harness` from a stale install — if you moved the
project, reinstall from the new directory.

## Layout

| file | contains |
| --- | --- |
| `providers.py` | the provider layer: ollama / openai / openrouter → chat model |
| `tools.py` | the workspace tools — knows nothing about models or the console |
| `harness.py` | the `Harness` agent loop and its console rendering |
| `main.py` | all command-line code: `chat`, `serve`, `init-provider` |
| `api.py` | FastAPI app implementing the OpenAI chat-completions contract |
| `providers.example.yaml` | reference template for `providers.yaml` |

`Harness.run_events()` is the loop as a stream of events (`step`, `token`, `usage`,
`message`, `tool`, `result`, `done`, `error`). `Harness.run()` renders those to the
terminal; the API serializes the same events as SSE. One loop, two front ends.

## Providers

**The default provider is ollama.** Run `harness` (or `harness serve`) with no
`--provider` and it uses ollama, configured from `.env` (`HARNESS_MODEL`,
`HARNESS_BASE_URL`, or the legacy `OLLAMA_MODEL` / `OLLAMA_HOST`).

Every other provider is selected explicitly with `--provider`, which accepts a
built-in name (`openai`, `openrouter`) or any name you define in `providers.yaml`
(see below). The config file is read **only** when `--provider` is given; without
it the harness never looks at the file.

Built-in providers and their defaults (overridable by flags / `.env`):

| provider | default model | default endpoint | key env var |
| --- | --- | --- | --- |
| `ollama` | `qwen3.8-coder:latest` | `http://127.0.0.1:11434` | none |
| `openai` | `gpt-4o-mini` | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| `openrouter` | `deepseek/deepseek-chat` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |

```bash
# default: ollama, from .env
harness

# OpenRouter
harness --provider openrouter --model deepseek/deepseek-chat

# OpenAI or any OpenAI-compatible server (vLLM, LM Studio, Groq, ...)
harness --provider openai --base-url http://localhost:8000/v1 --model llama-3.1-8b

# set keys in .env, or export them; HARNESS_API_KEY works for any provider
harness serve --provider openai
```

The `.env` file is read automatically — `.env` in the launch directory first, then
the project's own `.env` (copy `.env.example`). Flags always win over env vars.
`--num-ctx` only applies to ollama.

## Provider config file (`providers.yaml`)

Named providers are defined in a YAML file and used by name — no code changes
needed to add a provider. Each entry stores its own `base_url`, `api_key` and
default `model`; `--provider NAME` selects it, and `--model` overrides the
entry's default model for that run.

The file is **only read when `--provider` is passed**. Without `--provider` the
harness uses ollama from `.env` and never touches the config. There is no
`default` key: the default provider is always ollama.

Search order (first match wins): `--config PATH`, `$HARNESS_CONFIG`,
`./providers.yaml` in the launch directory, `~/.config/harness/providers.yaml`,
`%APPDATA%\harness\providers.yaml`.

```yaml
# providers.yaml
providers:
  openrouter:
    base_url: https://openrouter.ai/api/v1
    api_key: sk-or-v1-xxxxxxxxxxxxxxxx
    model: openai/gpt-5.5
  groq:
    base_url: https://api.groq.com/openai/v1   # custom names default to the
    api_key: gsk_xxxx                          # openai protocol (OpenAI-compatible)
    model: llama-3.3-70b-versatile
    num_ctx: 16000
  local:
    kind: ollama                          # switch the wire protocol if needed
    base_url: http://127.0.0.1:11434
    model: qwen3
```

```bash
harness --provider openrouter                    # config defaults: endpoint, key, model
harness --provider openrouter --model openai/gpt-5.5   # override the model for this run
harness --provider groq --config ~/code/providers.yaml
```

### Initializing a provider from the command line

The fastest way to add OpenRouter (or any provider) is to let the harness write
the entry for you:

```bash
# fully flagged — writes providers.yaml, no prompts
harness init-provider --name openrouter \
  --base-url https://openrouter.ai/api/v1 \
  --api-key sk-or-v1-xxxxxxxxxxxxxxxx \
  --model openai/gpt-5.5

# interactive — asks for each field and validates the entry
harness init-provider

# a second provider merges into the same file
harness init-provider --name groq --base-url https://api.groq.com/openai/v1 \
  --api-key gsk_xxxx --model llama-3.3-70b-versatile
```

Pass `--config PATH` to target a specific file (default: `./providers.yaml`).
Names that match a built-in (openrouter, openai, ollama) inherit its wire
protocol; custom names (groq, …) default to the openai protocol unless you set
`--kind`. The reference template ships as `providers.example.yaml`.

Precedence per field: **flag > config entry > env var > built-in default**. A
config entry named `openrouter`/`openai`/`ollama` inherits that built-in's wire
protocol; any other name defaults to the `openai` protocol (any OpenAI-compatible
endpoint) unless the entry sets `kind`. Unknown provider names error with the
list of known providers.

OpenAI-compatible tool calling is used for `openai`/`openrouter`; Ollama's own
tool calling for `ollama`. The stored conversation is portable across providers.

## Entering a task

A task is one line by default. For anything longer:

- `"""` on its own line opens a multi-line block; paste freely, then close it with
  another `"""`. Pasted lines are consumed by the block instead of each firing as its
  own task.
- `/file <path>` runs a task stored in a file — the practical way to re-run a prompt.
- `harness --file prompt.md` does the same from the shell, then exits.

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

Off by default: `spawn_agent` is not in the tool list and the delegation rules are
not in the system prompt unless you pass `--subagents` or type `/subagents`. The model
cannot decide on its own to start delegating.

`spawn_agent(task, files)` runs one fresh agent to completion, sequentially, and
returns its report. A sub-agent starts with an empty conversation, cannot spawn agents
of its own, and its tokens are added to the parent's totals.

Ownership is enforced, not requested:

- Every sub-agent declares the files it owns. Claiming a file another sub-agent already
  owns is refused.
- A sub-agent that tries to write, patch or restore a file outside its own list gets an
  error telling it to report the needed change instead.

Sequential by design — one local instance gains nothing from concurrent requests, and
serial execution keeps the file-ownership guarantee simple.

## Permissions

Six tools mutate the workspace or the machine: `write_file`, `apply_patch`,
`run_command`, `run_tests`, `git_restore`, `install_dependency`.

- `--mode ask` (chat default) — confirm each call
- `--mode deny` (serve default) — read-only; refusals go back to the model, which
  turns the agent into a reviewer rather than a dead end
- `--mode allow` — no prompting

In the REPL, `/allow`, `/ask` and `/deny` switch mode mid-session; `/help` lists
every command.

## API

Server-side agent: one request runs the whole task. Tools execute in the server's
workspace, so the client needs no filesystem access and no loop of its own. The
server uses the same provider resolution as the CLI.

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

`usage` sums input/output tokens across every step of the agent loop, so a single
request that took four model calls reports the cost of all four. The same totals
ride on the final SSE chunk.

### Serving safely

`serve` binds to `127.0.0.1` and defaults to `--mode deny`. `--mode allow` on a
reachable interface is remote code execution by design: pair it with `--api-key`
(or `HARNESS_API_KEY`) and keep the bind address local.

```bash
harness serve --mode allow --api-key "$(openssl rand -hex 16)" --workspace ~/code/project
```

## Notes

- The REPL prompt shows how full the context window is (`ask 34%>`); `/stats` prints
  context, steps, tokens in/out and model time for the session.
- The model must support tool calling (`qwen3-coder`, `qwen2.5-coder`, `llama3.1+`,
  `mistral-nemo`, and OpenAI-style providers generally). If tool calls only appear on
  the non-streaming path, use `--no-stream`.
- `--num-ctx` defaults to 200,000; set it to what your model actually supports
  (Ollama's own default is 2048, which silently truncates the tool schemas and looks
  like the model ignoring its tools). `OLLAMA_HOST` is often a bind address
  (`0.0.0.0:11434`); it is rewritten to a dialable one automatically.