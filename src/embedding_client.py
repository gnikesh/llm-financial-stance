import os
import torch
from typing import Union, List


class EmbeddingClient:
    """
    Embedding client that supports two backends:

    1. Local server (default): calls a local HTTP server on port 8000
       Usage: EmbeddingClient(url="http://localhost:8000")

    2. Nautilus API: calls qwen3-embedding (or other models) via the
       OpenAI-compatible Nautilus endpoint using NAUTILUS_API_KEY
       Usage: EmbeddingClient(backend="nautilus", model="qwen3-embedding")
    """

    def __init__(
        self,
        url: str = "http://localhost:8000",
        backend: str = "local",
        model: str = "qwen3-embedding",
        nautilus_url: str = "https://ellm.nrp-nautilus.io/v1",
    ):
        self.backend = backend
        self.model = model

        if backend == "nautilus":
            import openai
            self.client = openai.OpenAI(
                api_key=os.getenv("NAUTILUS_API_KEY"),
                base_url=nautilus_url,
            )
        else:
            # Local server mode
            import requests
            self._requests = requests
            self.url = url.rstrip("/")

    def get_embeddings(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """Get embeddings and return as torch tensor."""
        if self.backend == "nautilus":
            return self._get_nautilus_embeddings(texts)
        else:
            return self._get_local_embeddings(texts)

    def _get_nautilus_embeddings(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """Get embeddings from Nautilus API (OpenAI-compatible endpoint)."""
        # OpenAI API expects a list of strings (or a single string)
        input_texts = [texts] if isinstance(texts, str) else texts

        response = self.client.embeddings.create(
            model=self.model,
            input=input_texts,
        )

        # Extract embedding vectors and sort by index
        embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        tensor = torch.tensor(embeddings, dtype=torch.float32)

        # Handle single text case: return 1D tensor
        if isinstance(texts, str):
            tensor = tensor.squeeze(0)

        return tensor

    def _get_local_embeddings(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """Get embeddings from local HTTP server."""
        try:
            response = self._requests.post(
                f"{self.url}/embed",
                json={"texts": texts},
                timeout=60,
            )

            if response.status_code != 200:
                raise Exception(f"API error: {response.status_code} - {response.text}")

            result = response.json()
            embeddings = result["embeddings"]
            tensor = torch.tensor(embeddings, dtype=torch.float32)

            # Handle single text case
            if isinstance(texts, str):
                tensor = tensor.squeeze(0)

            return tensor

        except self._requests.exceptions.RequestException as e:
            raise Exception(f"Connection error: {e}")
        except Exception as e:
            raise Exception(f"Client error: {e}")

    def health_check(self):
        """Check server health (local backend only)."""
        if self.backend == "nautilus":
            # Quick test with a short string
            try:
                self.get_embeddings("test")
                return {"status": "ok", "backend": "nautilus", "model": self.model}
            except Exception as e:
                return {"status": "error", "error": str(e)}
        try:
            response = self._requests.get(f"{self.url}/health", timeout=10)
            return response.json() if response.status_code == 200 else None
        except:
            return None


if __name__ == "__main__":
    # Test with Nautilus backend
    client = EmbeddingClient(backend="nautilus", model="qwen3-embedding")
    x = client.get_embeddings(["This is a test sentence.", "This is another test sentence."])
    print(f"Shape: {x.shape}")
    print(f"Tensor:\n{x}")
