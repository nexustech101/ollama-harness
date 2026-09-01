"""OpenAI-compatible HTTP API for the harness.

One POST to /v1/chat/completions runs the whole agent loop server-side: tools
execute in the server's workspace, the model's tokens and tool activity stream
back as `reasoning_content` deltas, and the final answer arrives as `content`.
Any OpenAI client works against it unmodified.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from providers import DEFAULT_NUM_CTX
from harness import Harness, fmt_args


@dataclass
class ServerConfig:
    """Everything the API needs that a request cannot be trusted to supply."""

    model: str
    base_url: str
    workspace: Path
    mode: str = "deny"          # deny (read-only) / ask (unusable headless) / allow
    num_ctx: int = DEFAULT_NUM_CTX
    max_steps: int = 0          # 0 = no limit; retry_limit ends stuck runs
    retry_limit: int = 5
    provider: str = "ollama"
    api_key: Optional[str] = None       # bearer auth for /v1 routes
    model_api_key: Optional[str] = None  # key the model call uses (config/env resolved)
    subagents: bool = False
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    models: tuple[str, ...] = ()    # curated model ids from the config entry


# ---------------------------------------------------------------------------
# Request / response bodies (unknown fields from OpenAI clients are ignored)
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str
    content: Optional[str] = ""


class ChatRequest(BaseModel):
    messages: list[Message]
    model: Optional[str] = None
    stream: bool = False
    temperature: float = 0.0
    max_steps: Optional[int] = None


def _completion(request_id: str, created: int, model: str, text: str,
                usage: dict[str, int]) -> dict[str, Any]:
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        # summed across every step of the agent loop, as reported by Ollama
        "usage": usage,
    }


def _usage(agent: Harness) -> dict[str, int]:
    totals = agent.stats
    return {
        "prompt_tokens": totals["input"],
        "completion_tokens": totals["output"],
        "total_tokens": totals["input"] + totals["output"],
    }


def _chunk(request_id: str, created: int, model: str, delta: dict[str, Any],
           finish: Optional[str] = None) -> str:
    body = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(body)}\n\n"


def create_app(config: ServerConfig) -> FastAPI:
    app = FastAPI(title="ollama-harness", version="0.1.0",
                  description="OpenAI-compatible API for a local coding agent.")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def authorize(authorization: Optional[str]) -> None:
        if not config.api_key:
            return
        if authorization != f"Bearer {config.api_key}":
            raise HTTPException(status_code=401, detail="invalid api key")

    def build_agent(request: ChatRequest) -> tuple[Harness, str]:
        """Replay the posted history into a fresh agent; return it and the query."""
        model = request.model or config.model
        if request.model and config.models and request.model not in config.models:
            raise HTTPException(
                status_code=400,
                detail=f"unknown model {request.model!r}. Available: "
                       f"{', '.join(config.models)}")
        agent = Harness(
            model=model,
            workspace=config.workspace,
            base_url=config.base_url,
            max_steps=request.max_steps if request.max_steps is not None else config.max_steps,
            retry_limit=config.retry_limit,
            provider=config.provider,
            api_key=config.model_api_key,
            subagents=config.subagents,
            mode=config.mode,
            temperature=request.temperature,
            num_ctx=config.num_ctx,
            stream=True,
        )
        for message in request.messages:
            content = message.content or ""
            if message.role == "system":
                agent.messages[0] = SystemMessage(f"{agent.messages[0].content}\n\n{content}")
            elif message.role == "user":
                agent.messages.append(HumanMessage(content))
            elif message.role == "assistant":
                agent.messages.append(AIMessage(content))
        if not isinstance(agent.messages[-1], HumanMessage):
            raise HTTPException(status_code=400, detail="the last message must have role 'user'")
        return agent, agent.messages.pop().content

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "provider": config.provider,
            "model": config.model,
            "workspace": str(config.workspace),
            "mode": config.mode,
        }

    @app.get("/v1/models")
    def models(authorization: Optional[str] = Header(None)) -> dict[str, Any]:
        authorize(authorization)
        ids = list(config.models) or [config.model]
        return {
            "object": "list",
            "data": [{
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "ollama-harness",
            } for model_id in ids],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatRequest, authorization: Optional[str] = Header(None)):
        authorize(authorization)
        agent, query = build_agent(request)
        model = request.model or config.model
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if not request.stream:
            answer = ""
            for event in agent.run_events(query):
                if event["type"] == "done":
                    answer = event["text"]
                elif event["type"] == "error":
                    raise HTTPException(status_code=502, detail=event["text"])
            return _completion(request_id, created, model, answer, _usage(agent))

        def stream() -> Iterator[str]:
            yield _chunk(request_id, created, model, {"role": "assistant", "content": ""})
            try:
                for event in agent.run_events(query):
                    kind = event["type"]
                    if kind == "token":
                        # the model's own output, including any <think> block
                        yield _chunk(request_id, created, model,
                                     {"reasoning_content": event["text"]})
                    elif kind == "tool":
                        yield _chunk(request_id, created, model, {"reasoning_content":
                                     f"\n→ {event['name']}  {fmt_args(event['args'])}\n"})
                    elif kind == "result":
                        yield _chunk(request_id, created, model,
                                     {"reasoning_content": f"  {event['summary']}\n"})
                    elif kind == "done":
                        yield _chunk(request_id, created, model, {"content": event["text"]})
                    elif kind == "error":
                        yield _chunk(request_id, created, model,
                                     {"content": f"\n[error] {event['text']}"})
            except Exception as e:  # a stream cannot change its status code
                yield _chunk(request_id, created, model,
                             {"content": f"\n[error] {type(e).__name__}: {e}"})
            final = _chunk(request_id, created, model, {}, finish="stop")
            body = json.loads(final[6:])
            body["usage"] = _usage(agent)          # clients with include_usage read this
            yield f"data: {json.dumps(body)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    return app
