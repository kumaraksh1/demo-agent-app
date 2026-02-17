"""OpenTelemetry configuration with Azure Monitor exporters for traces and metrics."""
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


def setup_telemetry(app=None):
    """Configure OpenTelemetry with Azure Monitor exporter."""
    connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    service_name = os.getenv("OTEL_SERVICE_NAME", "demo-agent-app")

    if not connection_string:
        print("Warning: APPLICATIONINSIGHTS_CONNECTION_STRING not set. Telemetry disabled.")
        return trace.get_tracer(service_name)

    resource = Resource.create({SERVICE_NAME: service_name})

    # Traces
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(AzureMonitorTraceExporter(connection_string=connection_string))
    )
    trace.set_tracer_provider(provider)

    # Metrics
    metric_reader = PeriodicExportingMetricReader(
        AzureMonitorMetricExporter(connection_string=connection_string),
        export_interval_millis=60000,
    )
    metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=[metric_reader]))

    if app:
        FastAPIInstrumentor.instrument_app(app)

    print(f"OpenTelemetry configured for service: {service_name}")
    return trace.get_tracer(service_name)


def get_tracer(name: str = "demo-agent-app"):
    """Get a tracer instance."""
    return trace.get_tracer(name)


# ── Metrics (GenAI semantic conventions) ───────────────────────────────

_meter = metrics.get_meter("gen_ai")


def get_token_usage_metric():
    """Histogram for gen_ai.client.token.usage (input/output token counts)."""
    return _meter.create_histogram(
        name="gen_ai.client.token.usage",
        description="Measures the number of input and output tokens used",
        unit="token",
    )


def get_tool_duration_metric():
    """Histogram for gen_ai.client.tool.duration (tool call duration in seconds)."""
    return _meter.create_histogram(
        name="gen_ai.client.tool.duration",
        description="Measures the duration of tool calls in seconds",
        unit="s",
    )
