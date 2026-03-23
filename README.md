# AI-IA Agentic (Autonomous Coding Teammate)

This project is a Streamlit app that runs an autonomous “plan → generate → test → patch” loop to implement a requested feature.

## Quick start

1. Create/activate a virtual environment:
   - `python -m venv .venv`
   - `source .venv/bin/activate` (macOS/Linux)
2. Install dependencies:
   - `pip install -r requirement.txt`
3. Configure API keys:
   - Copy `.env.example` to `.env`
   - Set `OPENAI_API_KEY` (required) and optionally `GEMINI_API_KEY` (fallback)

4. Run the app:
   - `streamlit run app.py`

Open the URL Streamlit prints (by default `http://localhost:8501`).

## Environment variables

See `.env.example` for placeholders.

- `OPENAI_API_KEY` (required for primary model)
- `GEMINI_API_KEY` (optional fallback; if OpenAI fails)

If `OPENAI_API_KEY` is missing or OpenAI calls fail, `tools/llm_client.py` falls back to the Gemini model `gemini-2.0-flash` using the Gemini REST API.

## Run tests

This repo includes example-generated tests under `examples/`.

- `python -m pytest -q`

## Notes

- The sandbox “local” mode runs `pytest` for the generated example project; Docker mode additionally runs inside a container (requires Docker).
- Generated example artifacts under `examples/` are ignored by git (`.gitignore`).

