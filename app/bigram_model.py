"""Bigram language model for the simple text generator.

Built from the Module 2 lecture code: counts how often each word follows
another word in the corpus, turns the counts into probabilities, and
generates new text by sampling the next word from those probabilities.
"""
import random
from collections import Counter, defaultdict


class BigramModel:
    def __init__(self, corpus):
        # corpus: a list of text strings
        bigram_counts = defaultdict(Counter)
        for text in corpus:
            words = text.lower().split()
            for current_word, next_word in zip(words, words[1:]):
                bigram_counts[current_word][next_word] += 1

        # turn raw counts into probabilities
        self.bigram_probs = {}
        for current_word, counter in bigram_counts.items():
            total = sum(counter.values())
            self.bigram_probs[current_word] = {
                next_word: count / total for next_word, count in counter.items()
            }

    def generate_text(self, start_word, length):
        """Generate `length` words starting from `start_word`."""
        word = start_word.lower()
        result = [word]
        for _ in range(length - 1):
            probs = self.bigram_probs.get(word)
            if not probs:
                break  # unknown word: stop early
            next_words = list(probs.keys())
            weights = list(probs.values())
            word = random.choices(next_words, weights=weights)[0]
            result.append(word)
        return " ".join(result)
