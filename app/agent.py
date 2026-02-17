"""
LangChain Agent with Azure OpenAI and OpenTelemetry tracing.
Uses OTel GenAI semantic conventions so Azure Monitor Agents (preview) blade works.
"""
import os
from typing import Optional
from langchain_openai import AzureChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
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


def _extract_token_usage(result, span):
    """Extract token usage from LangChain LLMResult and set span attributes + record metrics."""
    input_tokens = 0
    output_tokens = 0
    model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "unknown")

    # Try generation-level info first
    if result.generations and len(result.generations) > 0:
        gen = result.generations[0][0] if isinstance(result.generations[0], list) else result.generations[0]
        if hasattr(gen, 'generation_info') and gen.generation_info:
            info = gen.generation_info
            if 'token_usage' in info:
                input_tokens = info['token_usage'].get('prompt_tokens', 0)
                output_tokens = info['token_usage'].get('completion_tokens', 0)

    # Try llm_output (aggregated) – usually more reliable
    if hasattr(result, 'llm_output') and result.llm_output:
        if 'model_name' in result.llm_output:
            model_name = result.llm_output['model_name']
            span.set_attribute("gen_ai.response.model", model_name)
        if 'token_usage' in result.llm_output:
            usage = result.llm_output['token_usage']
            input_tokens = usage.get('prompt_tokens', 0)
            output_tokens = usage.get('completion_tokens', 0)

    # Set span attributes (for trace-level visibility)
    span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
    span.set_attribute("gen_ai.usage.output_tokens", output_tokens)

    # Record metrics (for Azure Monitor Token Consumption dashboards)
    common_attrs = {
        "gen_ai.request.model": model_name,
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


class TracedAzureChatOpenAI(AzureChatOpenAI):
    """Azure OpenAI chat model with OpenTelemetry GenAI semantic convention tracing."""

    def _create_chat_span(self):
        """Start a CLIENT span named 'chat <model>' per GenAI semconv."""
        model = self.deployment_name or "unknown"
        span = tracer.start_span(
            f"chat {model}",
            kind=SpanKind.CLIENT,
            attributes={
                "gen_ai.operation.name": "chat",
                "gen_ai.system": AZURE_OPENAI_PROVIDER,
                "gen_ai.provider.name": AZURE_OPENAI_PROVIDER,
                "gen_ai.request.model": model,
                "gen_ai.request.temperature": self.temperature or 0.7,
                "server.address": os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            },
        )
        return span

    def _generate(self, *args, **kwargs):
        span = self._create_chat_span()
        ctx = trace.set_span_in_context(span)
        token = trace.context_api.attach(ctx)
        try:
            result = super()._generate(*args, **kwargs)
            _extract_token_usage(result, span)
            return result
        except Exception as exc:
            span.set_attribute("error.type", type(exc).__name__)
            span.record_exception(exc)
            raise
        finally:
            span.end()
            trace.context_api.detach(token)

    async def _agenerate(self, *args, **kwargs):
        span = self._create_chat_span()
        ctx = trace.set_span_in_context(span)
        token = trace.context_api.attach(ctx)
        try:
            result = await super()._agenerate(*args, **kwargs)
            _extract_token_usage(result, span)
            return result
        except Exception as exc:
            span.set_attribute("error.type", type(exc).__name__)
            span.record_exception(exc)
            raise
        finally:
            span.end()
            trace.context_api.detach(token)


def create_agent() -> Optional[AgentExecutor]:
    """Create and configure the LangChain agent with Azure OpenAI."""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

    if not endpoint or not api_key:
        print("Warning: Azure OpenAI credentials not configured")
        return None

    llm = TracedAzureChatOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        deployment_name=deployment,
        api_version=api_version,
        temperature=0.7,
    )

    prompt = PromptTemplate.from_template(REACT_PROMPT)
    agent = create_react_agent(llm, all_tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=all_tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
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
