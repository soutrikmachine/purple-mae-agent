"""
A2A server entry point for the Purple MAE Agent.

Run locally:
    python main.py --host 0.0.0.0 --port 9009

Docker:
    docker run -p 9009:9009 rimodock/purple-mae-agent:latest

Health check (used by the gateway to know we're ready):
    curl http://localhost:9009/.well-known/agent-card.json

Important: This module imports from `a2a.server.apps`, which exists in the
a2a-sdk 0.3.x series. The 1.0 release removed those wrapper classes; we pin
to <1.0.0 in requirements.txt.
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Task,
    TaskState,
    UnsupportedOperationError,
)
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError

from negotiator import handle_negotiation_message

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("purple_mae.server")


class PurpleMAEExecutor(AgentExecutor):
    """Routes every A2A text message to the negotiator dispatcher."""

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        try:
            message_text = context.get_user_input() or ""
            context_id = context.context_id
            logger.info(
                "incoming context_id=%s len=%d head=%r",
                context_id,
                len(message_text),
                message_text[:200],
            )

            msg = context.message
            updater = None
            if msg:
                task = new_task(msg)
                await event_queue.enqueue_event(task)
                updater = TaskUpdater(event_queue, task.id, task.context_id)

            response = handle_negotiation_message(message_text)
            response_text = json.dumps(response)
            logger.info("response len=%d body=%s", len(response_text), response_text[:300])

            if updater is not None:
                await updater.update_status(
                    TaskState.completed,
                    new_agent_text_message(response_text, context_id=context_id),
                )

        except Exception as exc:
            logger.exception("Unhandled error processing request: %s", exc)
            # Safe fallback: emit a JSON object the green's parser can read as
            # a non-accepting response. Better than raising and getting nothing
            # back (which causes the green to retry then default to WALK).
            error_payload = {
                "accept": False,
                "reason": f"internal error: {exc}",
            }
            try:
                if "updater" in locals() and updater is not None:
                    await updater.update_status(
                        TaskState.completed,
                        new_agent_text_message(
                            json.dumps(error_payload),
                            context_id=context.context_id,
                        ),
                    )
            except Exception:
                logger.exception("Failed even to report the error")

    async def cancel(self, request: RequestContext, event_queue: EventQueue) -> Task | None:
        raise ServerError(error=UnsupportedOperationError())


def _agent_card(url: str) -> AgentCard:
    skill = AgentSkill(
        id="bargaining-negotiation",
        name="Meta-Game Bargaining Negotiation",
        description=(
            "Strategic negotiator for OpenSpiel bargaining. Hybrid policy: "
            "deterministic concession-schedule core (provably avoids M1-M5) "
            "with optional LLM and RL refinement layers."
        ),
        tags=["bargaining", "negotiation", "purple-agent", "mae", "agentbeats"],
        examples=[
            "Propose an item allocation given my valuations and BATNA.",
            "Accept or reject an incoming offer given my BATNA and the discount.",
        ],
    )
    return AgentCard(
        name=os.environ.get("AGENT_NAME", "Purple MAE Negotiator"),
        version="0.2.0",
        description=(
            "Purple agent for the AgentBeats x AgentX Meta-Game Bargaining Evaluator."
        ),
        url=url,
        preferred_transport="JSONRPC",
        protocol_version="0.3.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Purple MAE Agent A2A server")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("AGENT_PORT", os.environ.get("PORT", "9009"))),
    )
    parser.add_argument(
        "--card-url",
        default=None,
        help="Public URL advertised on the agent card (defaults to http://host:port/)",
    )
    args = parser.parse_args()

    base_url = (
        args.card_url
        or os.environ.get("AGENT_URL")
        or f"http://{args.host}:{args.port}/"
    )

    use_llm = os.environ.get("USE_LLM", "false").lower() in ("1", "true", "yes")
    use_rl = os.environ.get("USE_RL", "false").lower() in ("1", "true", "yes")
    provider = os.environ.get("LLM_PROVIDER", "openrouter")
    model = os.environ.get("LLM_MODEL", "google/gemini-2.0-flash")
    opening_aggro = os.environ.get("OPENING_AGGRESSIVENESS", "0.75")
    if use_llm:
        key_env = "OPENROUTER_API_KEY" if provider == "openrouter" else "ANTHROPIC_API_KEY"
        if not os.environ.get(key_env):
            logger.warning(
                "USE_LLM=true but %s is unset; LLM layer will fall back to deterministic.",
                key_env,
            )
    logger.info(
        "Starting Purple MAE Agent at %s (LLM=%s/%s/%s, RL=%s, aggressiveness=%s)",
        base_url, use_llm, provider, model, use_rl, opening_aggro,
    )

    executor = PurpleMAEExecutor()
    card = _agent_card(base_url)
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(agent_card=card, http_handler=handler)

    uvicorn.Server(
        uvicorn.Config(
            server.build(),
            host=args.host,
            port=args.port,
            log_level=os.environ.get("UVICORN_LOG_LEVEL", "info"),
        )
    ).run()


if __name__ == "__main__":
    main()
