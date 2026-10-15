from typing import Union

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.bigram_model import BigramModel
from app.classifier import ImageClassifier
from app.embedding_model import EmbeddingModel

app = FastAPI()

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]
bigram_model = BigramModel(corpus)

# spaCy word embedding model (loaded once at startup)
embedding_model = EmbeddingModel()

# CNN image classifier trained on CIFAR10 (loaded once at startup)
image_classifier = ImageClassifier()


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


class EmbeddingRequest(BaseModel):
    word: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


@app.post("/embed")
def embed_word(request: EmbeddingRequest):
    """Return the spaCy word embedding for the query word."""
    try:
        return embedding_model.get_embedding(request.word)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    """Classify an uploaded image with the CIFAR10 CNN.

    Returns the predicted class id, class name and confidence.
    """
    try:
        image_bytes = await file.read()
        return image_classifier.predict(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Could not classify image: {e}')
