# working_embedding_server.py
import os
import sys

# Set environment variables BEFORE any imports
os.environ["DISABLE_FLASH_ATTN"] = "1"
os.environ["FLASH_ATTENTION_SKIP_CUDA_BUILD"] = "TRUE"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Disable flash attention at the module level
import importlib.util
flash_attn_spec = importlib.util.find_spec("flash_attn")
if flash_attn_spec is not None:
    print("Warning: flash_attn found, but will be disabled")
    sys.modules['flash_attn'] = None

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import uvicorn
from typing import List, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variable
model = None

class EmbedRequest(BaseModel):
    texts: Union[str, List[str]]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model
    try:
        logger.info("Loading SentenceTransformer model...")

        # Import here to control the import process
        from sentence_transformers import SentenceTransformer

        # Force CPU mode to avoid CUDA issues
        # device = 'cpu'  # Force CPU for stability
        device = 'cuda' if torch.cuda.is_available() else device
        logger.info(f"Using device: {device}")

        # Load model with explicit settings to avoid flash attention
        model = SentenceTransformer(
            'all-mpnet-base-v2',
            device=device,
            cache_folder=None  # Use default cache
        )

        # Test the model
        test_embedding = model.encode("test", convert_to_tensor=True)
        logger.info(f"Model loaded successfully! Test embedding shape: {test_embedding.shape}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error("Check if sentence-transformers is properly installed")
        raise

    yield

    # Shutdown
    logger.info("Shutting down...")

# Create FastAPI app with lifespan
app = FastAPI(title="Working Embedding Server", lifespan=lifespan)

@app.post("/embed")
async def embed_texts(request: EmbedRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        texts = request.texts if isinstance(request.texts, list) else [request.texts]

        logger.info(f"Processing {len(texts)} text(s)")

        # Generate embeddings with explicit settings
        embeddings = model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False,
            batch_size=16,  # Smaller batch size for stability
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        # Convert to list for JSON serialization
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)

        embeddings_list = embeddings.cpu().numpy().tolist()

        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return {"embeddings": embeddings_list}

    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model": "all-mpnet-base-v2",
        "device": "cpu",
        "torch_version": torch.__version__,
        "flash_attn_disabled": True
    }

@app.get("/")
async def root():
    return {"message": "Embedding Server is running", "endpoints": ["/embed", "/health"]}

# ============ CLIENT CODE ============
# import requests
# import torch
# from typing import Union, List

# class WorkingEmbeddingClient:
#     def __init__(self, url: str = "http://localhost:8000"):
#         self.url = url.rstrip('/')

#     def get_embeddings(self, texts: Union[str, List[str]]) -> torch.Tensor:
#         """Get embeddings and return as torch tensor"""
#         try:
#             response = requests.post(
#                 f"{self.url}/embed",
#                 json={"texts": texts},
#                 timeout=60  # Increased timeout for CPU processing
#             )

#             if response.status_code != 200:
#                 raise Exception(f"API error: {response.status_code} - {response.text}")

#             result = response.json()
#             embeddings = result["embeddings"]

#             # Convert to torch tensor
#             tensor = torch.tensor(embeddings, dtype=torch.float32)

#             # Handle single text case
#             if isinstance(texts, str):
#                 tensor = tensor.squeeze(0)

#             return tensor

#         except requests.exceptions.RequestException as e:
#             raise Exception(f"Connection error: {e}")
#         except Exception as e:
#             raise Exception(f"Client error: {e}")

#     def health_check(self):
#         """Check server health"""
#         try:
#             response = requests.get(f"{self.url}/health", timeout=10)
#             return response.json() if response.status_code == 200 else None
#         except:
#             return None

# ============ TEST FUNCTION ============
# def test_embedding_server():
#     """Test the embedding server"""
#     print("Testing embedding server...")

#     client = WorkingEmbeddingClient()

#     # Check health
#     health = client.health_check()
#     if health:
#         print(f"✓ Server health: {health}")
#     else:
#         print("✗ Server not responding")
#         return False

#     try:
#         # Test single text
#         print("Testing single text...")
#         text_embd = client.get_embeddings("This is a test sentence")
#         print(f"✓ Single embedding shape: {text_embd.shape}")
#         print(f"✓ Type: {type(text_embd)}")

#         # Test multiple texts
#         print("Testing multiple texts...")
#         texts = ["First text", "Second text", "Third text"]
#         batch_embd = client.get_embeddings(texts)
#         print(f"✓ Batch embedding shape: {batch_embd.shape}")

#         print("✓ All tests passed!")
#         return True

#     except Exception as e:
#         print(f"✗ Test failed: {e}")
#         return False

if __name__ == "__main__":
    print("Starting embedding server...")
    # print("Note: Using CPU mode for maximum compatibility")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
