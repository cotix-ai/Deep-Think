import asyncio
from typing import List
from openai import AsyncOpenAI


class LLMService:
    def __init__(self, api_key: str, base_url: str):
        if not api_key or "YOUR_API_KEY" in api_key:
            raise ValueError(
                "API key is not configured. Please set the MOLE_API_KEY environment variable.")
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def get_chat_completions(self, prompt: str, model: str, n: int, max_tokens: int, temp: float, stop: List[str] | None = None) -> List[str]:
        if (stop is None):
            stop = ["\n"]
        try:
            response = await self.client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                n=n, max_tokens=max_tokens, temperature=temp, stop=stop,
            )
            return [
                choice.message.content.strip()  # type: ignore
                for choice in response.choices
            ]
        except Exception:
            await asyncio.sleep(2)
            return []
