from typing import Union

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.bigram_model import BigramModel
from app.classifier import ImageClassifier
from app.diffusion_generator import DiffusionGenerator
from app.ebm_generator import EBMGenerator
from app.embedding_model import EmbeddingModel
from app.generator import DigitGenerator

app = FastAPI()

corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]
bigram_model = BigramModel(corpus)

embedding_model = EmbeddingModel()

image_classifier = ImageClassifier()

digit_generator = DigitGenerator()

diffusion_generator = DiffusionGenerator()

ebm_generator = EBMGenerator()


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


class EmbeddingRequest(BaseModel):
    word: str


class GenerateImageRequest(BaseModel):
    seed: Union[int, None] = None


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


@app.post("/embed")
def embed_word(request: EmbeddingRequest):
    try:
        return embedding_model.get_embedding(request.word)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        return image_classifier.predict(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Could not classify image: {e}')


@app.post("/generate-image")
def generate_image(request: GenerateImageRequest):
    try:
        return digit_generator.generate(request.seed)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Could not generate image: {e}')


@app.post("/generate-diffusion")
def generate_diffusion(request: GenerateImageRequest):
    try:
        return diffusion_generator.generate(request.seed)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Could not generate image: {e}')


@app.post("/generate-ebm")
def generate_ebm(request: GenerateImageRequest):
    try:
        return ebm_generator.generate(request.seed)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Could not generate image: {e}')
