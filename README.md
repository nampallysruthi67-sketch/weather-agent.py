# Weather Agent

An LLM that decides *when* to call tools and *which* ones — the agent loop written by hand, not hidden behind a framework.

**1 of 10** — part of a GenAI project series. FastAPI backend, vanilla JS frontend.

## What it demonstrates

- Tool/function calling with the Gemini API
- A manual agent loop: model → tool call → result → model → answer
- Tool results are ground truth; the model narrates, it does not invent numbers

## Run it

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # Windows: copy .env.example .env
# add your key to .env  (skip if this project needs no key)
uvicorn main:app --reload
```

Open http://127.0.0.1:8000

## Keys

| Key | Where to get it | Where it goes |
|---|---|---|
| `GEMINI_API_KEY` | https://aistudio.google.com/apikey (free) | `.env` |

The weather data comes from Open-Meteo, which needs no key.

## Stack

FastAPI · google-genai · Open-Meteo

## How it works

The model gets two tool declarations (`geocode`, `get_forecast`). It picks which to call and with what arguments. `main.py` executes the real function, feeds the result back, and loops until the model stops asking for tools. The loop is capped at 5 turns. Every step is returned to the UI so you can see the agent think.

---
MIT
