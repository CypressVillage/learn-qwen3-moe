"""Encode text with the byte-level BPE stored in a Qwen3 tokenizer.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import regex


class Qwen3Tokenizer:
    """A small, inference-only reader for Qwen3 byte-level BPE assets."""

    def __init__(
        self,
        vocabulary: dict[str, int],
        merges: list[str | list[str]],
        pretoken_pattern: str,
        special_tokens: dict[str, int],
    ) -> None:
        self.vocabulary = vocabulary
        self.tokens = {token_id: token for token, token_id in vocabulary.items()}
        self.special_tokens = special_tokens
        self.special_token_ids = frozenset(special_tokens.values())
        self._merge_ranks = {
            self._parse_merge(pair): rank for rank, pair in enumerate(merges)
        }
        self._pretokenizer = regex.compile(pretoken_pattern)
        self._special_pattern = (
            regex.compile(
                "|".join(
                    regex.escape(token)
                    for token in sorted(special_tokens, key=len, reverse=True)
                )
            )
            if special_tokens
            else None
        )
        self._byte_encoder, self._byte_decoder = _byte_alphabet()
        self._bpe_cache: dict[str, tuple[str, ...]] = {}

    @classmethod
    def from_file(cls, path: str | Path) -> "Qwen3Tokenizer":
        """Read the vocabulary, merge ranks, and special tokens."""

        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        model = raw.get("model")
        if not isinstance(model, dict) or model.get("type") != "BPE":
            raise ValueError("tokenizer.json must contain a BPE model")
        vocabulary = model.get("vocab")
        merges = model.get("merges")
        if not isinstance(vocabulary, dict) or not isinstance(merges, list):
            raise ValueError("tokenizer.json is missing its vocabulary or merges")
        if not all(
            isinstance(token, str) and isinstance(token_id, int)
            for token, token_id in vocabulary.items()
        ):
            raise ValueError("tokenizer vocabulary must map strings to integer IDs")

        special_tokens = {
            item["content"]: item["id"]
            for item in raw.get("added_tokens", [])
            if item.get("special") is True
        }
        full_vocabulary = {**vocabulary, **special_tokens}
        if len(set(full_vocabulary.values())) != len(full_vocabulary):
            raise ValueError("tokenizer IDs must be unique")
        return cls(
            full_vocabulary,
            merges,
            _find_split_pattern(raw.get("pre_tokenizer")),
            special_tokens,
        )

    @staticmethod
    def _parse_merge(pair: str | list[str]) -> tuple[str, str]:
        parts = pair.split(" ", 1) if isinstance(pair, str) else pair
        if len(parts) != 2:
            raise ValueError(f"invalid BPE merge: {pair!r}")
        return parts[0], parts[1]

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary)

    def token_to_id(self, token: str) -> int:
        try:
            return self.vocabulary[token]
        except KeyError as error:
            raise KeyError(f"unknown token: {token!r}") from error

    def id_to_token(self, token_id: int) -> str:
        try:
            return self.tokens[token_id]
        except KeyError as error:
            raise KeyError(f"unknown token ID: {token_id}") from error

    def encode(self, text: str) -> list[int]:
        """Turn text into the exact token IDs used to index model embeddings."""

        if self._special_pattern is None:
            return self._encode_text(text)

        token_ids: list[int] = []
        start = 0
        for match in self._special_pattern.finditer(text):
            token_ids.extend(self._encode_text(text[start : match.start()]))
            token_ids.append(self.special_tokens[match.group(0)])
            start = match.end()
        token_ids.extend(self._encode_text(text[start:]))
        return token_ids

    def _encode_text(self, text: str) -> list[int]:
        token_ids: list[int] = []
        for match in self._pretokenizer.finditer(text):
            byte_token = "".join(
                self._byte_encoder[byte] for byte in match.group(0).encode("utf-8")
            )
            pieces = self._bpe_cache.get(byte_token)
            if pieces is None:
                pieces = _merge_token(byte_token, self._merge_ranks)
                self._bpe_cache[byte_token] = pieces
            try:
                token_ids.extend(self.vocabulary[piece] for piece in pieces)
            except KeyError as error:
                raise ValueError(f"BPE produced an unknown token: {error.args[0]!r}") from error
        return token_ids

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = False,
        errors: str = "replace",
    ) -> str:
        """Reassemble token IDs into UTF-8 text."""

        output: list[str] = []
        byte_buffer = bytearray()

        def flush_bytes() -> None:
            if byte_buffer:
                output.append(byte_buffer.decode("utf-8", errors=errors))
                byte_buffer.clear()

        for token_id in token_ids:
            token = self.id_to_token(token_id)
            if token_id in self.special_token_ids:
                flush_bytes()
                if not skip_special_tokens:
                    output.append(token)
                continue
            try:
                byte_buffer.extend(self._byte_decoder[character] for character in token)
            except KeyError as error:
                raise ValueError(
                    f"token contains a character outside the byte alphabet: {token!r}"
                ) from error
        flush_bytes()
        return "".join(output)


def _byte_alphabet() -> tuple[dict[int, str], dict[str, int]]:
    """Map all 256 byte values to visible Unicode characters and back."""

    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))
    code_points = byte_values.copy()
    extra = 0
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            code_points.append(256 + extra)
            extra += 1
    encoder = dict(zip(byte_values, map(chr, code_points), strict=True))
    return encoder, {character: byte for byte, character in encoder.items()}


def _merge_token(
    token: str, merge_ranks: dict[tuple[str, str], int]
) -> tuple[str, ...]:
    """Repeatedly apply the highest-priority merge still present in one token."""

    pieces = tuple(token)
    while len(pieces) > 1:
        adjacent_pairs = zip(pieces, pieces[1:])
        best_pair = min(adjacent_pairs, key=lambda pair: merge_ranks.get(pair, float("inf")))
        if best_pair not in merge_ranks:
            break

        merged: list[str] = []
        index = 0
        while index < len(pieces):
            if index + 1 < len(pieces) and pieces[index : index + 2] == best_pair:
                merged.append("".join(best_pair))
                index += 2
            else:
                merged.append(pieces[index])
                index += 1
        pieces = tuple(merged)
    return pieces


def _find_split_pattern(pretokenizer: object) -> str:
    if not isinstance(pretokenizer, dict):
        raise ValueError("tokenizer.json is missing its pre-tokenizer")
    if pretokenizer.get("type") == "Split":
        pattern = pretokenizer.get("pattern")
        if isinstance(pattern, dict) and isinstance(pattern.get("Regex"), str):
            return pattern["Regex"]
    for child in pretokenizer.get("pretokenizers", []):
        try:
            return _find_split_pattern(child)
        except ValueError:
            pass
    raise ValueError("tokenizer.json does not contain a regex Split pre-tokenizer")
