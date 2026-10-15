# sps_genai

Simple text generation FastAPI project (Module 3 class activity), extended with
a spaCy word embedding endpoint for Assignment 1 and a CNN image classifier
endpoint for Assignment 2.

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
- `POST /classify` — CNN image classification (CIFAR10, 10 classes). Upload an
  image file; returns `{"predicted_class": 3, "class_name": "cat", "confidence": 0.87}`.
  Train the model first: `uv run python train_cifar10.py` (saves `app/cifar10_cnn.pt`).

## Docker (optional)

```bash
docker build -t sps-genai .
docker run -p 8000:80 sps-genai
```
