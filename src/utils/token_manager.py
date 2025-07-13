import tiktoken


class TokenManager:
    def __init__(self, model="gpt-4o-mini") -> None:
        self.encoding = tiktoken.encoding_for_model(model)

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        count_tokens = self.count_tokens(text)
        tokens = self.encoding.encode(text)
        if count_tokens <= max_tokens:
            return text

        truncated = tokens[:max_tokens]
        return self.encoding.decode(truncated) + "\n\n[ДАННЫЕ ОБРЕЗАНЫ]"
