"""
LangChain Agent with Azure OpenAI and OpenTelemetry tracing.
Uses a LangChain callback handler to capture token usage and emit
OTel GenAI semantic convention spans + metrics so the Azure Monitor
Gen AI (preview) dashboard works end-to-end.
"""
import os
from typing import Any, Optional
from uuid import UUID
from langchain_openai import AzureChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from opentelemetry import trace
from opentelemetry.trace import SpanKind
from app.telemetry import get_tracer, get_token_usage_metric
from app.tools import all_tools

tracer = get_tracer("langchain-agent")
token_usage_histogram = get_token_usage_metric()

AZURE_OPENAI_PROVIDER = "azure.ai.openai"

# ReAct agent prompt template
REACT_PROMPT = """You are a helpful assistant with access to tools. Use them when needed.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""


class OpenTelemetryCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that creates OTel GenAI spans and records
    token-usage metrics for every LLM call."""

    def __init__(self, model_name: str = "unknown"):
        super().__init__()
        self.model_name = model_name
        # Map run_id -> span so on_llm_end can close the right span
        self._spans: dict[UUID, Any] = {}

    # ── LLM start ──────────────────────────────────────────────────────
    def on_llm_start(self, serialized: dict, prompts: list[str], *,
                     run_id: UUID, **kwargs) -> None:
        span = tracer.start_span(
            f"chat {self.model_name}",
            kind=SpanKind.CLIENT,
            attributes={
                "gen_ai.operation.name": "chat",
                "gen_ai.system": AZURE_OPENAI_PROVIDER,
                "gen_ai.provider.name": AZURE_OPENAI_PROVIDER,
                "gen_ai.request.model": self.model_name,
                "server.address": os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            },
        )
        ctx = trace.set_span_in_context(span)
        token = trace.context_api.attach(ctx)
        self._spans[run_id] = (span, token)

    # ── LLM end (success) ─────────────────────────────────────────────
    def on_llm_end(self, response: LLMResult, *, run_id: UUID,
                   **kwargs) -> None:
        entry = self._spans.pop(run_id, None)
        if entry is None:
            return
        span, ctx_token = entry

        input_tokens = 0
        output_tokens = 0
        model = self.model_name

        # --- Strategy 1: llm_output.token_usage (set by agenerate) ---
        if response.llm_output:
            usage = response.llm_output.get("token_usage") or {}
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            m = response.llm_output.get("model_name")
            if m:
                model = m

        # --- Strategy 2: AIMessage.response_metadata.token_usage ---
        if input_tokens == 0 and response.generations:
            gen_list = response.generations[0]
            if gen_list:
                gen = gen_list[0] if isinstance(gen_list, list) else gen_list
                msg = getattr(gen, "message", None)
                if msg is not None:
                    # response_metadata (dict) contains token_usage
                    meta = getattr(msg, "response_metadata", None) or {}
                    tu = meta.get("token_usage") or {}
                    if tu:
                        input_tokens = tu.get("prompt_tokens", 0)
                        output_tokens = tu.get("completion_tokens", 0)
                    m = meta.get("model_name")
                    if m:
                        model = m

        # --- Strategy 3: AIMessage.usage_metadata (pydantic or dict) ---
        if input_tokens == 0 and response.generations:
            gen_list = response.generations[0]
            if gen_list:
                gen = gen_list[0] if isinstance(gen_list, list) else gen_list
                msg = getattr(gen, "message", None)
                if msg is not None:
                    um = getattr(msg, "usage_metadata", None)
                    if um is not None:
                        # Could be a dict or a pydantic UsageMetadata object
                        if isinstance(um, dict):
                            input_tokens = um.get("input_tokens", 0)
                            output_tokens = um.get("output_tokens", 0)
                        else:
                            input_tokens = getattr(um, "input_tokens", 0)
                            output_tokens = getattr(um, "output_tokens", 0)

        # Set span attributes
        span.set_attribute("gen_ai.response.model", model)
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)

        # Record histogram metrics for Token Consumption dashboards
        common_attrs = {
            "gen_ai.request.model": model,
            "gen_ai.system": AZURE_OPENAI_PROVIDER,
            "gen_ai.operation.name": "chat",
        }
        if input_tokens:
            token_usage_histogram.record(
                input_tokens,
                {**common_attrs, "gen_ai.token.type": "input"},
            )
        if output_tokens:
            token_usage_histogram.record(
                output_tokens,
                {**common_attrs, "gen_ai.token.type": "output"},
            )

        span.end()
        trace.context_api.detach(ctx_token)

    # ── LLM error ─────────────────────────────────────────────────────
    def on_llm_error(self, error: BaseException, *, run_id: UUID,
                     **kwargs) -> None:
        entry = self._spans.pop(run_id, None)
        if entry is None:
            return
        span, ctx_token = entry
        span.set_attribute("error.type", type(error).__name__)
        span.record_exception(error)
        span.end()
        trace.context_api.detach(ctx_token)


def create_agent() -> Optional[AgentExecutor]:
    """Create and configure the LangChain agent with Azure OpenAI."""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

    if not endpoint or not api_key:
        print("Warning: Azure OpenAI credentials not configured")
        return None

    otel_handler = OpenTelemetryCallbackHandler(model_name=deployment)

    llm = AzureChatOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        deployment_name=deployment,
        api_version=api_version,
        temperature=0.7,
        callbacks=[otel_handler],
    )

    prompt = PromptTemplate.from_template(REACT_PROMPT)
    agent = create_react_agent(llm, all_tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=all_tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        callbacks=[otel_handler],
    )
    return agent_executor


async def run_agent(query: str) -> dict:
    """Run the agent with a query and return the response with tracing."""
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
    # Wrap the whole agent invocation in an 'invoke_agent' span
    with tracer.start_as_current_span(
        f"invoke_agent react_agent",
        kind=SpanKind.INTERNAL,
        attributes={
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.system": AZURE_OPENAI_PROVIDER,
            "gen_ai.provider.name": AZURE_OPENAI_PROVIDER,
            "gen_ai.request.model": deployment,
            "gen_ai.agent.name": "react_agent",
            "gen_ai.agent.id": os.getenv("AGENT_RESOURCE_ID", "agent_12345"),
        },
    ) as span:
        agent = create_agent()
        if not agent:
            return {
                "success": False,
                "error": "Agent not configured. Check Azure OpenAI credentials.",
                "response": None,
            }

        try:
            result = await agent.ainvoke({"input": query})
            output = result.get("output", "")
            span.set_attribute("agent.output", output[:500])
            return {
                "success": True,
                "response": output,
                "error": None,
            }
        except Exception as e:
            error_msg = str(e)
            span.set_attribute("error.type", type(e).__name__)
            span.record_exception(e)
            return {
                "success": False,
                "error": error_msg,
                "response": None,
            }
