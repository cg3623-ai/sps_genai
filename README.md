# sps_genai

This is my FastAPI project. It started as the Module 3 class activity (a simple
bigram text generator), and I kept adding to it for each assignment: a spaCy
embedding endpoint for Assignment 1, a CNN image classifier for Assignment 2,
and a GAN digit generator for Assignment 3.

## Setup

```bash
uv sync
uv run python -m spacy download en_core_web_md
```

Make sure you get `en_core_web_md` (or `lg`), not `en_core_web_sm` — the `sm`
one doesn't come with word vectors, so the embed endpoint won't work with it.

## How to run it

```bash
uv run fastapi dev app/main.py
```

Then go to http://127.0.0.1:8000/docs — you can try all the endpoints from
that page.

## Endpoints

- `GET /` — just a health check, returns hello world
- `POST /generate` — bigram text generation, body: `{"start_word": "the", "length": 20}`
- `POST /embed` — gives you the spaCy embedding of a word, body: `{"word": "king"}`
- `POST /classify` — upload an image and the CIFAR10 CNN tells you which of the
  10 classes it is, e.g. `{"predicted_class": 3, "class_name": "cat", "confidence": 0.87}`.
  The model weights are in `app/cifar10_cnn.pt` (train it yourself with
  `uv run python train_cifar10.py` if the file is missing).
- `POST /generate-image` — the GAN generates a handwritten digit (MNIST).
  Body: `{"seed": 42}` — the seed is optional, use it if you want the same
  digit again. You get back a base64-encoded 28x28 PNG, like
  `{"image_base64": "...", "seed": 42, "format": "png", "size": "28x28"}`.
  Weights are in `app/mnist_gan.pt` (train with `uv run python train_gan.py`,
  takes a while on CPU).

## Docker (if you want to run it that way)

```bash
docker build -t sps-genai .
docker run -p 8000:80 sps-genai
```
