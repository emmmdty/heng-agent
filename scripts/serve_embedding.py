# -*- coding: utf-8 -*-
"""Qwen3-Embedding 参考服务端（OpenAI 兼容 /v1/embeddings）

与 `scripts/serve_reranker.py` 对称：让检索链路不依赖外部付费网关即可完整复现。
Qwen3-Embedding 用 **last-token pooling + L2 归一化**（不是均值池化），
配 Qdrant 的 COSINE 距离。

依赖（不进主 pyproject，属评测/开发侧）：torch、transformers、fastapi、uvicorn。

用法：
    python scripts/serve_embedding.py --model-dir <Qwen3-Embedding-0.6B 路径> --port 11436
    export EMBEDDING_BASE_URL=http://127.0.0.1:11436/v1 EMBEDDING_API_KEY=local-no-auth
    export EMBEDDING_MODEL=qwen3-embedding-0.6b EMBEDDING_DIM=1024

**必须返回 `usage` 字段**（见下方注释）：本仓自己的 `OpenAIEmbeddingClient` 不读它，
但 AgentScope 的 `rag.KnowledgeBase` 会直接取 `usage.total_tokens`。少了它，
商品向量索引一切正常、品类知识库却静默建库失败——同一个服务喂两个消费者，
只有其中一个挂，排查时极容易往错的方向找。
"""
from __future__ import annotations

import argparse
from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoTokenizer

BATCH_SIZE = 8


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
    ]


class EmbeddingRequest(BaseModel):
    model: str = Field(min_length=1)
    input: str | list[str] = Field(min_length=1)


class EmbeddingObject(BaseModel):
    object: str = "embedding"
    index: int
    embedding: list[float]


class Usage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingResponse(BaseModel):
    object: str = "list"
    model: str
    data: list[EmbeddingObject]
    usage: Usage = Usage()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = AutoModel.from_pretrained(args.model_dir, device_map="cuda")
    app.state.tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    app.state.model.eval()
    yield
    app.state.model = None


app = FastAPI(title="qwen3-embedding openai-compatible embedding server", lifespan=lifespan)


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
def embed(request: EmbeddingRequest) -> EmbeddingResponse:
    texts = [request.input] if isinstance(request.input, str) else list(request.input)
    if not texts:
        raise ValueError("input must not be empty")
    model = app.state.model
    tokenizer = app.state.tokenizer
    results: list[EmbeddingObject] = []
    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt",
            ).to(model.device)
            output = model(**encoded)
            vectors = F.normalize(
                last_token_pool(output[0], encoded["attention_mask"]), p=2, dim=1
            ).cpu().tolist()
            for offset, vector in enumerate(vectors):
                results.append(EmbeddingObject(index=start + offset, embedding=vector))
    # usage 必须返回：OpenAI embeddings 规范里它是响应的一部分，
    # AgentScope 的 rag.KnowledgeBase 会直接取 usage.total_tokens，
    # 缺了就是 'NoneType' object has no attribute 'total_tokens'。
    total = sum(len(tokenizer(t, truncation=True, max_length=2048)['input_ids']) for t in texts)
    return EmbeddingResponse(model=request.model, data=results,
                             usage=Usage(prompt_tokens=total, total_tokens=total))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=11436)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
