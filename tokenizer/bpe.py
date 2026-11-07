from __future__ import annotations

from collections import Counter, defaultdict
import heapq
import json
from pathlib import Path
from typing import Callable, Iterable

import regex


# GPT-2-style pre-token boundaries, not an existing BPE implementation.
PATTERN = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
Pair = tuple[int, int]


def _byte_unicode_map():
    """Map each byte to a distinct visible Unicode character (GPT-2 style)."""
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    mapping = {byte: chr(byte) for byte in visible}
    extra = 256
    for byte in range(256):
        if byte not in mapping:
            mapping[byte] = chr(extra)
            extra += 1
    return mapping


BYTE_TO_UNICODE = _byte_unicode_map()


def unicode_token(raw: bytes) -> str:
    return "".join(BYTE_TO_UNICODE[byte] for byte in raw)


def merge_pair(tokens: tuple[int, ...], pair: Pair, new_id: int) -> tuple[int, ...]:
    """Replace non-overlapping occurrences, scanning from left to right."""
    result = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
            result.append(new_id)
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return tuple(result)


def pair_counts(tokens: tuple[int, ...]) -> Counter:
    return Counter(zip(tokens, tokens[1:]))


class BPETokenizer:
    def __init__(self, pattern: str = PATTERN):
        self.pattern = pattern
        self._regex = regex.compile(pattern)
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges: list[Pair] = []
        self.merge_counts: list[int] = []
        self._ranks: dict[Pair, int] = {}
        self.training_stats: dict = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def pretokenize(self, text: str):
        for match in self._regex.finditer(text):
            yield match.group().encode("utf-8")

    def train(
        self,
        texts: Iterable[str],
        vocab_size: int = 16000,
        min_frequency: int = 2,
        progress: Callable[[dict], None] | None = None,
    ) -> dict:
        """Fit on training documents only; repeated calls reset learned state.

        Each unique pre-token is stored once with its frequency. An inverted
        index maps pairs to affected words; a lazy heap picks the most frequent
        pair. Only affected words are rescanned after each merge. Ties are
        resolved by token ID, making training deterministic.
        """
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")
        if min_frequency < 1:
            raise ValueError("min_frequency must be positive")
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = []
        self.merge_counts = []
        self._ranks = {}
        self.training_stats = {}
        counts = Counter()
        num_documents = num_bytes = 0
        for text in texts:
            if not isinstance(text, str):
                raise TypeError("Training documents must be strings")
            counts.update(self.pretokenize(text))
            num_documents += 1
            num_bytes += len(text.encode("utf-8"))
            if progress and num_documents % 100_000 == 0:
                progress({"stage": "counting", "documents": num_documents,
                          "utf8_bytes": num_bytes, "unique_pretokens": len(counts)})
        if not counts:
            raise ValueError("The training corpus is empty")

        words = [tuple(piece) for piece in counts]
        frequencies = list(counts.values())
        totals = Counter()
        containing = defaultdict(set)
        for word_id, word in enumerate(words):
            for pair, count in pair_counts(word).items():
                totals[pair] += count * frequencies[word_id]
                containing[pair].add(word_id)
        heap = [(-count, *pair) for pair, count in totals.items()]
        heapq.heapify(heap)
        if progress:
            progress({"stage": "counted", "documents": num_documents,
                      "utf8_bytes": num_bytes, "unique_pretokens": len(words)})

        while self.vocab_size < vocab_size:
            while heap:
                negative_count, left, right = heapq.heappop(heap)
                pair = (left, right)
                if totals.get(pair, 0) == -negative_count:
                    break
            else:
                break
            count = -negative_count
            if count < min_frequency:
                break
            new_id = len(self.vocab)
            self.vocab[new_id] = self.vocab[left] + self.vocab[right]
            self._ranks[pair] = len(self.merges)
            self.merges.append(pair)
            self.merge_counts.append(count)
            deltas = Counter()
            for word_id in sorted(containing[pair]):
                old = words[word_id]
                new = merge_pair(old, pair, new_id)
                before, after = pair_counts(old), pair_counts(new)
                frequency = frequencies[word_id]
                for changed_pair in before.keys() | after.keys():
                    deltas[changed_pair] += (
                        after[changed_pair] - before[changed_pair]
                    ) * frequency
                    if changed_pair not in after:
                        containing[changed_pair].discard(word_id)
                    elif changed_pair not in before:
                        containing[changed_pair].add(word_id)
                words[word_id] = new
            for changed_pair, delta in deltas.items():
                if not delta:
                    continue
                updated = totals[changed_pair] + delta
                if updated < 0:
                    raise AssertionError("Negative pair count")
                if updated:
                    totals[changed_pair] = updated
                    heapq.heappush(heap, (-updated, *changed_pair))
                else:
                    del totals[changed_pair]
                    containing.pop(changed_pair, None)
            # Bound memory used by obsolete heap entries.
            if len(heap) > 4 * len(totals) + 100_000:
                heap = [(-n, *p) for p, n in totals.items()]
                heapq.heapify(heap)
            if progress and (len(self.merges) == 1 or len(self.merges) % 500 == 0):
                progress({"stage": "merge", "merges": len(self.merges),
                          "vocab_size": self.vocab_size, "pair_frequency": count})

        self.training_stats = {
            "documents": num_documents, "utf8_bytes": num_bytes,
            "unique_pretokens": len(words), "pretoken_occurrences": sum(frequencies),
            "requested_vocab_size": vocab_size, "actual_vocab_size": self.vocab_size,
            "num_merges": len(self.merges), "min_frequency": min_frequency,
        }
        return self.training_stats.copy()

    def _encode_piece(self, piece: bytes) -> tuple[int, ...]:
        tokens = tuple(piece)
        while len(tokens) > 1:
            best_pair = None
            best_rank = len(self.merges)
            for pair in zip(tokens, tokens[1:]):
                rank = self._ranks.get(pair)
                if rank is not None and rank < best_rank:
                    best_pair, best_rank = pair, rank
            if best_pair is None:
                break
            tokens = merge_pair(tokens, best_pair, 256 + best_rank)
        return tokens

    def encode(self, text: str) -> list[int]:
        return [token for piece in self.pretokenize(text)
                for token in self._encode_piece(piece)]

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        return b"".join(self.vocab[token_id] for token_id in ids)

    def decode(self, ids: Iterable[int], errors: str = "strict") -> str:
        # A single byte token need not be valid UTF-8; decode the joined bytes.
        return self.decode_bytes(ids).decode("utf-8", errors=errors)

    def save(self, directory: str | Path):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        config = {
            "pattern": self.pattern, "vocab_size": self.vocab_size,
            "training_stats": self.training_stats,
        }
        vocabulary = [
            {"id": token_id,
             "token": unicode_token(raw),
             "bytes_hex": raw.hex()}
            for token_id, raw in sorted(self.vocab.items())
        ]
        merge_rules = [
            {"id": 256 + rank,
             "merge_tokens": [unicode_token(self.vocab[token]) for token in pair],
             "training_pair_frequency": self.merge_counts[rank]}
            for rank, pair in enumerate(self.merges)
        ]
        for name, value in [("tokenizer.json", config), ("vocab.json", vocabulary),
                            ("merges.json", merge_rules)]:
            destination = directory / name
            temporary = destination.with_suffix(".json.tmp")
            text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
            temporary.write_text(text, encoding="utf-8")
            temporary.replace(destination)

    @classmethod
    def load(cls, directory: str | Path) -> "BPETokenizer":
        directory = Path(directory)
        config = json.loads((directory / "tokenizer.json").read_text(encoding="utf-8"))
        model = cls(config["pattern"])
        vocabulary = json.loads((directory / "vocab.json").read_text(encoding="utf-8"))
        model.vocab = {row["id"]: bytes.fromhex(row["bytes_hex"]) for row in vocabulary}
        token_to_id = {row["token"]: row["id"] for row in vocabulary}
        rules = json.loads((directory / "merges.json").read_text(encoding="utf-8"))
        model.merges = [tuple(token_to_id[token] for token in rule["merge_tokens"]) for rule in rules]
        model.merge_counts = [rule["training_pair_frequency"] for rule in rules]
        model._ranks = {pair: rank for rank, pair in enumerate(model.merges)}
        return model
