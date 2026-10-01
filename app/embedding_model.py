"""Word embedding functionality built on spaCy.

This module is loaded once at API startup (see main.py) and exposes a
simple get_embedding(word) helper used by the POST /embed endpoint.

Requires the spaCy model 'en_core_web_md' (it ships with 300-dimensional
word vectors). Install it with:
    uv run python -m spacy download en_core_web_md
"""

import spacy


class EmbeddingModel:
    def __init__(self, model_name: str = "en_core_web_md"):
        # NOTE: en_core_web_sm does NOT ship word vectors, so use md or lg.
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            raise RuntimeError(
                f"spaCy model '{model_name}' is not installed. "
                f"Run: uv run python -m spacy download {model_name}"
            )

    def get_embedding(self, word: str) -> dict:
        """Return the embedding vector for a single query word.

        Returns a dict with the word, the vector as a list of floats,
        its dimension, and whether the word was found in the vocabulary.
        """
        word = word.strip()
        if not word:
            raise ValueError("word must be a non-empty string")

        token = self.nlp(word)[0]  # embedding of the first token
        vector = token.vector

        return {
            "word": word,
            "embedding": vector.tolist(),
            "dimension": int(vector.shape[0]),
            "has_vector": bool(token.has_vector),
        }
