from __future__ import annotations

import json
from collections.abc import Iterable, Iterator

import requests


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url: str, llm_model: str, embed_model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.llm_model = llm_model
        self.embed_model = embed_model

    def healthcheck(self) -> None:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaError(
                "Ollama is not reachable. Start it with `ollama serve` and pull the models."
            ) from exc

    def list_models(self) -> list[str]:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaError("Could not list local Ollama models.") from exc

        models = response.json().get("models", [])
        return [model.get("name", "") for model in models if model.get("name")]

    def embed(self, text: str) -> list[float]:
        payload = {"model": self.embed_model, "prompt": text}
        try:
            response = requests.post(f"{self.base_url}/api/embeddings", json=payload, timeout=60)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaError(f"Embedding request failed for model {self.embed_model}") from exc

        embedding = response.json().get("embedding")
        if not embedding:
            raise OllamaError("Ollama returned an empty embedding.")
        return [float(value) for value in embedding]

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    def generate(self, prompt: str, model: str | None = None) -> str:
        payload = {
            "model": model or self.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
            },
        }
        try:
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaError(f"Generation request failed for model {self.llm_model}") from exc

        answer = response.json().get("response", "").strip()
        if not answer:
            raise OllamaError("Ollama returned an empty answer.")
        return answer

    def generate_stream(self, prompt: str, model: str | None = None) -> Iterator[str]:
        payload = {
            "model": model or self.llm_model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
            },
        }
        try:
            with requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120,
                stream=True,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    payload = json.loads(line)
                    if "response" in payload:
                        yield payload["response"]
                    if payload.get("done"):
                        break
        except requests.RequestException as exc:
            raise OllamaError(f"Streaming request failed for model {model or self.llm_model}") from exc
