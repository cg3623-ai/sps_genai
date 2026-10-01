# sps_genai

Simple text generation FastAPI project (Module 3 class activity), extended with
a spaCy word embedding endpoint for Assignment 1.

## Setup

```bash
uv sync
uv run python -m spacy download en_core_web_md
```

Note: use `en_core_web_md` (or `lg`), not `en_core_web_sm` — the `sm` model
ships no word vectors.

## Run

```bash
uv run fastapi dev app/main.py
```

Open http://127.0.0.1:8000/docs for the interactive API docs.

## Endpoints

- `GET /` — health check
- `POST /generate` — bigram text generation, body: `{"start_word": "the", "length": 20}`
- `POST /embed` — spaCy word embedding, body: `{"word": "king"}`

## Docker (optional)

```bash
docker build -t sps-genai .
docker run -p 8000:80 sps-genai
```
