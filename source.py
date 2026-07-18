import os, json
import httpx
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-2.5-flash"

app = FastAPI(title="Weather Agent")


# ---------------------------------------------------------------- real tools
def geocode(place: str) -> dict:
    """Turn a place name into coordinates."""
    r = httpx.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": place, "count": 1},
        timeout=15,
    )
    hits = r.json().get("results")
    if not hits:
        return {"error": f"No place found matching {place!r}"}
    h = hits[0]
    return {
        "name": h["name"],
        "country": h.get("country"),
        "latitude": h["latitude"],
        "longitude": h["longitude"],
    }


def get_forecast(latitude: float, longitude: float, days: int = 3) -> dict:
    """Current conditions plus a daily forecast."""
    r = httpx.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "forecast_days": min(max(days, 1), 7),
            "timezone": "auto",
        },
        timeout=15,
    )
    return r.json()


TOOLS = {"geocode": geocode, "get_forecast": get_forecast}

DECLARATIONS = types.Tool(
    function_declarations=[
        {
            "name": "geocode",
            "description": "Convert a place name into latitude and longitude. Call this first.",
            "parameters": {
                "type": "object",
                "properties": {"place": {"type": "string", "description": "City or place name"}},
                "required": ["place"],
            },
        },
        {
            "name": "get_forecast",
            "description": "Get current weather and daily forecast for coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                    "days": {"type": "integer", "description": "1-7, default 3"},
                },
                "required": ["latitude", "longitude"],
            },
        },
    ]
)

SYSTEM = (
    "You are a weather agent. Use the tools to get real data before answering. "
    "Never guess a temperature or condition. If a tool returns an error, say so plainly. "
    "Answer in 2-4 sentences with the units the data gives you."
)


class Ask(BaseModel):
    question: str


@app.post("/api/ask")
def ask(body: Ask):
    """The agent loop, written out so you can watch it run."""
    contents = [types.Content(role="user", parts=[types.Part(text=body.question)])]
    trace = []

    for turn in range(5):
        resp = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                tools=[DECLARATIONS], system_instruction=SYSTEM
            ),
        )
        parts = resp.candidates[0].content.parts or []
        calls = [p.function_call for p in parts if p.function_call]

        # No tool calls left -> the model is answering.
        if not calls:
            return {"answer": resp.text or "(no answer)", "trace": trace}

        contents.append(resp.candidates[0].content)

        results = []
        for call in calls:
            fn = TOOLS.get(call.name)
            args = dict(call.args or {})
            out = fn(**args) if fn else {"error": f"unknown tool {call.name}"}
            trace.append({"turn": turn + 1, "tool": call.name, "args": args, "result": out})
            results.append(
                types.Part.from_function_response(name=call.name, response={"result": out})
            )

        contents.append(types.Content(role="user", parts=results))

    return {"answer": "Gave up after 5 tool turns.", "trace": trace}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
