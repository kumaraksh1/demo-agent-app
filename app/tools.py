"""Custom tools for the LangChain agent.

Tool call tracing (spans + duration metrics) is handled by the
OpenTelemetryCallbackHandler in agent.py via on_tool_start / on_tool_end,
so tools themselves contain only business logic.
"""
import random
from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Perform mathematical calculations. Input should be a valid math expression like '2 + 2' or '25 * 4'."""
    try:
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        return str(eval(expression))
    except Exception as e:
        return f"Error evaluating expression: {e}"


@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location. Returns temperature, conditions, and humidity."""
    weather_data = {
        "seattle": {"temp": 52, "condition": "Rainy", "humidity": 85},
        "new york": {"temp": 45, "condition": "Cloudy", "humidity": 60},
        "los angeles": {"temp": 72, "condition": "Sunny", "humidity": 40},
        "miami": {"temp": 82, "condition": "Partly Cloudy", "humidity": 75},
        "chicago": {"temp": 38, "condition": "Windy", "humidity": 55},
        "denver": {"temp": 48, "condition": "Clear", "humidity": 30},
    }
    data = weather_data.get(
        location.lower(),
        {
            "temp": random.randint(30, 85),
            "condition": random.choice(["Sunny", "Cloudy", "Rainy", "Clear"]),
            "humidity": random.randint(30, 90),
        },
    )
    return f"Weather in {location}: {data['temp']}°F, {data['condition']}, Humidity: {data['humidity']}%"


@tool
def web_search(query: str) -> str:
    """Search the web for information. Returns relevant search results."""
    return "\n".join([
        f"Result 1: Information about '{query}' from Wikipedia - A comprehensive overview of the topic.",
        f"Result 2: Latest news about '{query}' - Recent developments and updates.",
        f"Result 3: Expert analysis on '{query}' - In-depth research and findings.",
    ])


all_tools = [calculator, get_weather, web_search]
