import re
from dataclasses import dataclass


MARKER = "CONTEXT_INSERTION_729bd50f9b"
MC = "Answer the question from the context. Return only the letter A, B, C, or D."
QA = "Answer the question using a short exact span from the context in its original language. Output only that span."
SEPARATOR = "\n\n"


@dataclass(frozen=True)
class Piece:
    passage_id: str
    source: str
    left: int
    right: int

    @property
    def text(self):
        return self.source[self.left:self.right]


class Prompt:
    def __init__(self, tokenizer, variant):
        self.tokenizer = tokenizer
        multiple_choice = "options" in variant
        instruction = MC if multiple_choice else QA
        self.user_text = f"{instruction}\n\nContext:\n{MARKER}\n\nQuestion: {variant['question']}"
        if multiple_choice:
            self.user_text += "\n" + "\n".join(f"{letter}. {option}" for letter, option in
                                               zip("ABCD", variant["options"], strict=True))
        self.user_text += "\nAnswer:"
        messages = [{"role": "user", "content": self.user_text}]
        self.shell = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if self.shell.count(MARKER) != 1:
            raise ValueError("The chat template must preserve the context slot exactly once")
        self.before, self.after = self.shell.split(MARKER)

    def render(self, pieces):
        return self.before + SEPARATOR.join(piece.text for piece in pieces) + self.after

    def encode(self, text, offsets=False):
        return self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=offsets,
                              truncation=False, verbose=False)

    def count(self, pieces):
        return len(self.encode(self.render(pieces))["input_ids"])

    def describe(self, pieces, gold_id, spans):
        text = self.render(pieces)
        encoded = self.encode(text, offsets=True)
        offsets = encoded["offset_mapping"]
        cursor = len(self.before)
        locations = []
        span_locations = {}
        for piece in pieces:
            end = cursor + len(piece.text)
            hits = [i for i, (left, right) in enumerate(offsets) if left < right and left < end and right > cursor]
            locations.append({"passage_id": piece.passage_id, "source_start": piece.left,
                              "source_end": piece.right, "source_chars": len(piece.source),
                              "prompt_char_start": cursor, "prompt_char_end": end,
                              "token_start": hits[0] if hits else None,
                              "token_end": hits[-1] + 1 if hits else None})
            if piece.passage_id == gold_id:
                for index, span in enumerate(spans):
                    source_end = span["start"] + len(span["text"])
                    if piece.left <= span["start"] and source_end <= piece.right:
                        start = cursor + span["start"] - piece.left
                        stop = cursor + source_end - piece.left
                        hits = [i for i, (left, right) in enumerate(offsets)
                                if left < right and left < stop and right > start]
                        span_locations[index] = {"annotation_index": index,
                                                 "prompt_char_start": start, "prompt_char_end": stop,
                                                 "token_start": hits[0] if hits else None,
                                                 "token_end": hits[-1] + 1 if hits else None}
            cursor = end + len(SEPARATOR)
        gold_pieces = [piece for piece in locations if piece["passage_id"] == gold_id]
        ranges = sorted((piece["source_start"], piece["source_end"]) for piece in gold_pieces)
        covered = 0
        for start, end in ranges:
            if start <= covered:
                covered = max(covered, end)
        gold_length = gold_pieces[0]["source_chars"] if gold_pieces else None
        state = "complete" if gold_length is not None and covered == gold_length else (
            "partial" if gold_pieces else "absent")
        full_ids = sorted({piece["passage_id"] for piece in locations
                           if piece["source_start"] == 0 and piece["source_end"] == piece["source_chars"]})
        return {"prompt": text, "input_tokens": len(encoded["input_ids"]), "retained": locations,
                "complete_passage_ids": full_ids, "passage_state": state,
                "first_evidence_token": min((piece["token_start"] for piece in gold_pieces
                                             if piece["token_start"] is not None), default=None),
                "answer_span_retained": bool(span_locations) if spans else None,
                "retained_answer_spans": list(span_locations.values())}


def whole(passage_id, text):
    return Piece(passage_id, text, 0, len(text))


def fit_prefix(prompt, pieces, budget, fits=None):
    fits = fits or (lambda added: prompt.count(added) <= budget)
    kept = []
    for piece in pieces:
        if fits(kept + [piece]):
            kept.append(piece)
            continue
        offsets = prompt.encode(piece.text, offsets=True)["offset_mapping"]
        candidates = sorted({right for left, right in offsets if left < right})
        low, high = 0, len(candidates)
        while low < high:
            mid = (low + high) // 2
            trial = Piece(piece.passage_id, piece.source, piece.left, piece.left + candidates[mid])
            if fits(kept + [trial]):
                low = mid + 1
            else:
                high = mid
        if low:
            kept.append(Piece(piece.passage_id, piece.source, piece.left, piece.left + candidates[low - 1]))
        return kept
    return kept


def prefix(prompt, pieces, budget):
    return fit_prefix(prompt, pieces, budget)


def head_tail(prompt, pieces, budget):
    if prompt.count(pieces) <= budget:
        return pieces
    if len(pieces) == 1:
        piece = pieces[0]
        offsets = prompt.encode(piece.text, offsets=True)["offset_mapping"]
        ends = sorted({right for left, right in offsets if left < right})
        starts = sorted({left for left, right in offsets if left < right})
        low, high = 0, min(len(ends), len(starts)) + 1
        keep = []
        while low < high:
            width = (low + high) // 2
            if width == 0:
                trial = []
            elif ends[width - 1] >= starts[-width]:
                trial = pieces
            else:
                trial = [Piece(piece.passage_id, piece.source, 0, ends[width - 1]),
                         Piece(piece.passage_id, piece.source, starts[-width], piece.right)]
            if prompt.count(trial) <= budget:
                keep, low = trial, width + 1
            else:
                high = width
        return keep
    half = prompt.count([]) + (budget - prompt.count([])) // 2
    front = []
    for piece in pieces:
        if prompt.count(front + [piece]) > half:
            break
        front.append(piece)
    back = []
    for piece in reversed(pieces[len(front):]):
        if prompt.count(front + [piece] + back) > budget:
            break
        back.insert(0, piece)
    return front + back


def pack_ranked(prompt, pieces, ranked_ids, budget):
    selected = set()
    lookup = {piece.passage_id: piece for piece in pieces}
    for passage_id in ranked_ids:
        if passage_id not in lookup:
            raise ValueError(f"Unknown candidate {passage_id}")
        trial = selected | {passage_id}
        if prompt.count([piece for piece in pieces if piece.passage_id in trial]) <= budget:
            selected = trial
    return [piece for piece in pieces if piece.passage_id in selected]


def span_window(prompt, piece, spans, budget):
    span = spans[0]
    left, right = span["start"], span["start"] + len(span["text"])
    keep = Piece(piece.passage_id, piece.source, left, right)
    if prompt.count([keep]) > budget:
        return None
    cuts = [(mark.start(), mark.end()) for mark in re.finditer(r"\S+", piece.text)]
    low, high = 0, len(cuts) + 1
    before = [start for start, _ in cuts if start < left]
    after = [end for _, end in cuts if end > right]
    while low < high:
        wide = (low + high) // 2
        a = before[max(0, len(before) - wide)] if before and wide else left
        b = after[min(len(after), wide) - 1] if after and wide else right
        trial = Piece(piece.passage_id, piece.source, a, b)
        if prompt.count([trial]) <= budget:
            keep, low = trial, wide + 1
        else:
            high = wide
    return [keep]
