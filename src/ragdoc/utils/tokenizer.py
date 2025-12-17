from typing import Protocol, runtime_checkable

import tiktoken


@runtime_checkable
class Tokenizer(Protocol):
    def __call__(self, text: str) -> list[int]: ...

    def decode(self, tokens: list[int]) -> str: ...

    def truncate(self, text: str, max_tokens: int) -> str: ...

    def count(self, text: str) -> int: ...


class GPTTokenizer:
    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def __call__(self, text: str) -> list[int]:
        return self.encoding.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self.encoding.decode(tokens)

    def truncate(self, text: str, max_tokens: int) -> str:
        return self.decode(self(text)[:max_tokens])

    def count(self, text: str) -> int:
        return len(self(text))


class TransformerTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)

    def truncate(self, text: str, max_tokens: int) -> str:
        return self.tokenizer.decode(self.tokenizer.encode(text)[:max_tokens])

    def count(self, text: str) -> int:
        return len(self(text))


class RerankerTokenizer:  # implements Tokenizer
    def __init__(self, architecture: str = "BAAI/bge-reranker-v2-m3"):
        try:
            import transformers
        except ImportError as exc:
            raise ImportError(
                "RerankerTokenizer requires the 'tokenizers' extra: pip install 'ragdoc[tokenizers]'"
            ) from exc
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(architecture)

    def __call__(self, text: str) -> list[int]:
        return self.tokenizer(text).get("input_ids", [])[1:-1]  # tokenizer adds <s> text </s>.

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)

    def truncate(self, text: str, max_tokens: int) -> str:
        return self.decode(self(text)[:max_tokens])

    def count(self, text: str) -> int:
        return len(self(text))


class MaxTokenizer:  # implements Tokenizer
    def __init__(self, tokenizer: list[Tokenizer]):
        self.tokenizers = tokenizer
        self.last_tokenizer = None

    def __call__(self, text: str) -> list[int]:
        max_tokens = []
        for t in self.tokenizers:
            tokens = t(text)
            if len(tokens) > len(max_tokens):
                self.last_tokenizer = t
                max_tokens = tokens
        return max_tokens

    def decode(self, tokens: list[int]) -> str:
        assert self.last_tokenizer is not None, "No tokenizer was used"
        return self.last_tokenizer.decode(tokens)

    def truncate(self, text: str, max_tokens: int) -> str:
        return self.decode(self(text)[:max_tokens])

    def count(self, text: str) -> int:
        return len(self(text))
