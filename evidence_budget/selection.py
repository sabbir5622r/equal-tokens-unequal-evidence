import random
import time
import unicodedata
from collections import Counter

from .common import stable_id


def normalized(text):
    return " ".join("".join(char if unicodedata.category(char)[0] in "LMN" else " "
                            for char in unicodedata.normalize("NFKC", text).casefold()).split())


def grams(text, width=5):
    return {text[i:i + width] for i in range(max(0, len(text) - width + 1))}


def overlap(a, b):
    return len(a & b) / max(1, len(a | b))


class Candidates:
    def __init__(self, rows):
        self.pool = {}
        for row in rows:
            self.pool[row["cluster"]] = {language: variant["text"]
                                         for language, variant in row["variants"].items()}
        self.cleaned = {key: {language: normalized(text) for language, text in variants.items()}
                        for key, variants in self.pool.items()}
        self.grams = {key: {language: grams(text) for language, text in variants.items()}
                      for key, variants in self.cleaned.items()}

    def select(self, row, limit, seed, order):
        gold_id = row["cluster"]
        ranked = sorted(set(self.pool) - {gold_id})
        random.Random(int(stable_id(seed, order, row["id"])[:16], 16)).shuffle(ranked)
        clues = {language: normalized(variant["options"]["ABCD".index(variant["label"])])
                 for language, variant in row["variants"].items()}
        chosen = []
        reasons = Counter()
        for candidate in ranked:
            rejected = None
            for language, clue in clues.items():
                text = self.cleaned[candidate][language]
                if len(clue) >= 5 and f" {clue} " in f" {text} ":
                    rejected = "literal_gold_answer"
                    break
                candidate_grams = self.grams[candidate][language]
                if any(text == self.cleaned[existing][language] or
                       overlap(candidate_grams, self.grams[existing][language]) >= 0.35
                       for existing in [gold_id, *chosen]):
                    rejected = "near_duplicate"
                    break
            if rejected:
                reasons[rejected] += 1
                continue
            chosen.append(candidate)
            if len(chosen) == limit:
                return chosen, dict(reasons)
        raise ValueError(f"{row['id']}: need {limit} screened distractors, found {len(chosen)}")


class Retriever:
    def __init__(self, model_id, revision, device):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModel.from_pretrained(model_id, revision=revision).eval().to(device)
        self.documents = {}

    def encode(self, texts):
        torch = self.torch
        tokens = self.tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
        if tokens["input_ids"].shape[1] > 8192:
            raise ValueError("BGE-M3 input exceeds its supported length")
        with torch.inference_mode():
            encoded = self.model(**tokens.to(self.device)).last_hidden_state[:, 0]
            return torch.nn.functional.normalize(encoded.float(), dim=1).cpu()

    def build(self, candidates, batch=8):
        started = time.perf_counter()
        keys = [(language, key) for key, variants in sorted(candidates.pool.items())
                for language in sorted(variants)]
        for start in range(0, len(keys), batch):
            chunk = keys[start:start + batch]
            vectors = self.encode([candidates.pool[key][language] for language, key in chunk])
            self.documents.update(zip(chunk, vectors, strict=True))
        return {"index_seconds": time.perf_counter() - started, "indexed_passages": len(keys)}

    def rank(self, language, question, ids):
        started = time.perf_counter()
        query = self.encode([question])[0]
        documents = self.torch.stack([self.documents[(language, passage_id)] for passage_id in ids])
        scores = (documents @ query).tolist()
        ranked = sorted(zip(ids, scores, strict=True), key=lambda item: (-item[1], item[0]))
        return [passage_id for passage_id, _ in ranked], time.perf_counter() - started
