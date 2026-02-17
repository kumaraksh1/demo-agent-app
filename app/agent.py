"""LangChain Agent with Azure OpenAI and OpenTelemetry tracing.

disable_streaming=True is required on AzureChatOpenAI so LangChain uses
the non-streaming code path, which populates llm_output.token_usage.
The streaming path builds LLMResult via generate_from_stream() and
always leaves llm_output empty.
"""
import os
import time
from typing import Optional
from uuid import UUID

from langchain_openai import AzureChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from opentelemetry import trace
from opentelemetry.trace import SpanKind
from opentelemetry.trace import StatusCode

from app.telemetry import get_tracer, get_token_usage_metric, get_tool_duration_metric
from app.tools import all_tools

tracer = get_tracer("langchain-agent")
token_usage_histogram = get_token_usage_metric()
tool_duration_histogram = get_tool_duration_metric()

AZURE_OPENAI_PROVIDER = "azure.ai.openai"

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
    """LangChain callback that emits OTel GenAI spans, token-usage metrics,
    and tool-call duration metrics."""

    def __init__(self, model_name: str = "unknown"):
        super().__init__()
        self.model_name = model_name
        self._spans: dict[UUID, trace.Span] = {}   # run_id -> span
        self._tool_times: dict[UUID, float] = {}    # run_id -> start_time

    # -- LLM callbacks ---------------------------------------------------

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
        self._spans[run_id] = span

    def on_llm_end(self, response: LLMResult, *, run_id: UUID,
                   **kwargs) -> None:
        span = self._spans.pop(run_id, None)
        if span is None:
            return

        input_tokens = 0
        output_tokens = 0
        model = self.model_name

        # Extract tokens from llm_output.token_usage (populated when
        # disable_streaming=True on AzureChatOpenAI).
        if response.llm_output:
            usage = response.llm_output.get("token_usage") or {}
            input_tokens = usage.get("prompt_tokens", 0) or 0
            output_tokens = usage.get("completion_tokens", 0) or 0
            model = response.llm_output.get("model_name") or model

        span.set_attribute("gen_ai.response.model", model)
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)

        # Record histogram metric for Azure Monitor Gen AI dashboard
        common = {
            "gen_ai.request.model": model,
            "gen_ai.system": AZURE_OPENAI_PROVIDER,
            "gen_ai.operation.name": "chat",
        }
        if input_tokens:
            token_usage_histogram.record(
                input_tokens, {**common, "gen_ai.token.type": "input"},
            )
        if output_tokens:
            token_usage_histogram.record(
                output_tokens, {**common, "gen_ai.token.type": "output"},
            )

        span.end()

    def on_llm_error(self, error: BaseException, *, run_id: UUID,
                     **kwargs) -> None:
        span = self._spans.pop(run_id, None)
        if span is None:
            return
        span.set_status(StatusCode.ERROR, str(error))
        span.set_attribute("error.type", type(error).__name__)
        span.record_exception(error)
        span.end()

    # -- Tool callbacks (duration tracking) ------------------------------

    def on_tool_start(self, serialized: dict, input_str: str, *,
                      run_id: UUID, **kwargs) -> None:
        tool_name = serialized.get("name", "unknown")
        self._tool_times[run_id] = time.perf_counter()
        span = tracer.start_span(
            f"execute_tool {tool_name}",
            kind=SpanKind.INTERNAL,
            attributes={
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": tool_name,
                "gen_ai.tool.type": "function",
            },
        )
        self._spans[run_id] = span

    def on_tool_end(self, output: str, *, run_id: UUID, **kwargs) -> None:
        start = self._tool_times.pop(run_id, None)
        span = self._spans.pop(run_id, None)
        if span is None:
            return

        # Record tool call duration
        if start is not None:
            duration_s = time.perf_counter() - start
            # Extract tool name from span name "execute_tool <name>"
            span_name = span.name if hasattr(span, "name") else ""
            tool_name = span_name.replace("execute_tool ", "") if span_name.startswith("execute_tool ") else "unknown"
            tool_duration_histogram.record(
                duration_s,
                {
                    "gen_ai.tool.name": tool_name,
                    "gen_ai.system": AZURE_OPENAI_PROVIDER,
                    "gen_ai.operation.name": "execute_tool",
                },
            )

        span.end()

    def on_tool_error(self, error: BaseException, *, run_id: UUID,
                      **kwargs) -> None:
        self._tool_times.pop(run_id, None)
        span = self._spans.pop(run_id, None)
        if span is None:
            return
        span.set_status(StatusCode.ERROR, str(error))
        span.set_attribute("error.type", type(error).__name__)
        span.record_exception(error)
        span.end()


def create_agent() -> Optional[AgentExecutor]:
    """Create and configure the LangChain agent with Azure OpenAI."""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

    if not endpoint or not api_key:
        return None

    otel_handler = OpenTelemetryCallbackHandler(model_name=deployment)

    llm = AzureChatOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        deployment_name=deployment,
        api_version=api_version,
        temperature=0.7,
        disable_streaming=True,  # Required: non-streaming populates token_usage
        callbacks=[otel_handler],
    )

    prompt = PromptTemplate.from_template(REACT_PROMPT)
    agent = create_react_agent(llm, all_tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=all_tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        callbacks=[otel_handler],
    )


async def run_agent(query: str) -> dict:
    """Run the agent with a query and return the response with tracing."""
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")

    with tracer.start_as_current_span(
        "invoke_agent react_agent",
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
            return {"success": False, "error": "Agent not configured.", "response": None}

        try:
            result = await agent.ainvoke({"input": query})
            output = result.get("output", "")
            span.set_attribute("agent.output", output[:500])
            return {"success": True, "response": output, "error": None}
        except Exception as e:
            span.set_attribute("error.type", type(e).__name__)
            span.record_exception(e)
            return {"success": False, "error": str(e), "response": None}