"""FastAPI app that exposes a simple LLM-style generation endpoint.

For the purpose of this lab, we focus on:

- A clean, minimal API surface
- Integration with the MockVLLMEngine (KV cache + batching)
- CPU-only execution
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .vllm_mock import MockVLLMEngine


app = FastAPI(
    title="LLM Scale Lab — G-Ready Edition API",
    description="Minimal LLM-style API with KV cache & batching mock.",
    version="0.1.0",
)

engine = MockVLLMEngine(max_batch_size=8)


class HealthResponse(BaseModel):
    status: str


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="User prompt text.")
    max_new_tokens: int = Field(16, ge=1, le=256)


class GenerateResponse(BaseModel):
    request_id: int
    text: str


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    request_id = engine.submit(prompt=req.prompt, max_new_tokens=req.max_new_tokens)
    # For simplicity, we process the batch immediately here.
    results = engine.process_next_batch()
    # Find the result corresponding to this request_id.
    for r in results:
        if r.request_id == request_id:
            return GenerateResponse(request_id=request_id, text=r.text)
    # In a more complex system we'd handle async behavior; here it should not happen.
    return GenerateResponse(request_id=request_id, text="no result")