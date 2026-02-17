"""
OpenTelemetry configuration with GenAI semantic conventions for Azure Monitor.
"""
import os
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from azure.monitor.opentelemetry.exporter import (
    AzureMonitorTraceExporter,
    AzureMonitorMetricExporter,
)
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor


# GenAI Semantic Convention attribute names (OTel GenAI semconv)
class GenAIAttributes:
    """OpenTelemetry Semantic Conventions for Generative AI."""
    # Provider / system
    SYSTEM = "gen_ai.system"  # legacy compat
    PROVIDER_NAME = "gen_ai.provider.name"

    # Request attributes
    REQUEST_MODEL = "gen_ai.request.model"
    REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    REQUEST_TOP_P = "gen_ai.request.top_p"

    # Response attributes
    RESPONSE_ID = "gen_ai.response.id"
    RESPONSE_MODEL = "gen_ai.response.model"
    RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"

    # Usage
    USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

    # Operation
    OPERATION_NAME = "gen_ai.operation.name"

    # Tool attributes
    TOOL_NAME = "gen_ai.tool.name"
    TOOL_DESCRIPTION = "gen_ai.tool.description"
    TOOL_TYPE = "gen_ai.tool.type"

    # Agent
    AGENT_NAME = "gen_ai.agent.name"


def setup_telemetry(app=None):
    """Configure OpenTelemetry with Azure Monitor exporter."""
    connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    service_name = os.getenv("OTEL_SERVICE_NAME", "demo-agent-app")
    
    if not connection_string:
        print("Warning: APPLICATIONINSIGHTS_CONNECTION_STRING not set. Telemetry disabled.")
        return trace.get_tracer(service_name)
    
    # Create resource with service name
    resource = Resource.create({SERVICE_NAME: service_name})
    
    # Create tracer provider
    provider = TracerProvider(resource=resource)
    
    # Configure Azure Monitor trace exporter
    exporter = AzureMonitorTraceExporter(connection_string=connection_string)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    
    # Set as global tracer provider
    trace.set_tracer_provider(provider)

    # Configure Azure Monitor metric exporter + MeterProvider
    metric_exporter = AzureMonitorMetricExporter(connection_string=connection_string)
    metric_reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=60000)
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)
    
    # Instrument FastAPI if app is provided
    if app:
        FastAPIInstrumentor.instrument_app(app)
    
    print(f"OpenTelemetry configured for service: {service_name}")
    return trace.get_tracer(service_name)


def get_tracer(name: str = "demo-agent-app"):
    """Get a tracer instance."""
    return trace.get_tracer(name)


def get_token_usage_metric():
    """Return a Histogram instrument for GenAI token usage.

    Metric name follows the OpenTelemetry GenAI semantic conventions so that
    Azure Monitor's Gen AI dashboards can pick it up.
    """
    meter = metrics.get_meter("gen_ai")
    return meter.create_histogram(
        name="gen_ai.client.token.usage",
        description="Measures the number of input and output tokens used",
        unit="token",
    )
