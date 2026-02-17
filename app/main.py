"""FastAPI application for the Demo Agent Web App."""
import os

from dotenv import load_dotenv
load_dotenv()  # no-op when .env is absent (production)

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.telemetry import setup_telemetry, get_tracer
from app.agent import run_agent

app = FastAPI(
    title="Demo Agent Web App",
    description="LangChain Agent with Azure OpenAI and OpenTelemetry tracing",
    version="1.0.0",
)

templates = Jinja2Templates(directory="app/templates")
tracer = setup_telemetry(app)


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    success: bool
    response: str | None
    error: str | None


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the chat web UI."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message through the agent."""
    with tracer.start_as_current_span("chat_endpoint") as span:
        span.set_attribute("gen_ai.operation.name", "chat_request")
        span.set_attribute("chat.message_length", len(request.message))
        
        result = await run_agent(request.message)
        
        span.set_attribute("chat.success", result["success"])
        
        return ChatResponse(
            success=result["success"],
            response=result["response"],
            error=result["error"]
        )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": os.getenv("OTEL_SERVICE_NAME", "demo-agent-app"),
        "version": "1.0.0"
    }


@app.get("/config")
async def config():
    """Configuration status endpoint (for debugging)."""
    return {
        "azure_openai_configured": bool(os.getenv("AZURE_OPENAI_ENDPOINT")),
        "app_insights_configured": bool(os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")),
        "service_name": os.getenv("OTEL_SERVICE_NAME", "demo-agent-app"),
    }
