# -*- coding: utf-8 -*-
"""Qwen3-Reranker 参考服务端（/rerank 协议，配 app/infrastructure/rerank/http_reranker.py）

仓库里原先只有 reranker 的**客户端**，没有服务端，导致 `embedding_rerank` 档从来没被
真正验证过——评测里它和 `embedding_only` 跑出逐位相同的数字，报告脚注写着
"未配置 RERANKER_BASE_URL"。一个从未真正跑起来的档位不该出现在能力清单里，
所以把服务端补进仓库，让二阶段召回可复现。

依赖（不进主 pyproject，属评测/开发侧）：torch、transformers、fastapi、uvicorn。

用法：
    python scripts/serve_reranker.py --model-dir <Qwen3-Reranker-0.6B 路径> --port 11437
    export RERANKER_BASE_URL=http://127.0.0.1:11437 RERANKER_MODEL=qwen3-reranker-0.6b

Qwen3-Reranker 是 causal LM 形态的重排器，不是常规 cross-encoder 分类头：
把 (instruction, query, document) 拼成一段对话，取最后一个位置上 "yes" / "no"
两个 token 的 logit 做二分类 softmax，yes 的概率即相关度。
"""
from __future__ import annotations

import argparse

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

_INSTRUCT = "Given a shopper query, judge whether the product document satisfies it."
_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the "
    'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
_MAX_LEN = 1024
_BATCH = 16


class RerankRequest(BaseModel):
    model: str = ""
    query: str
    documents: list[str]


app = FastAPI(title="qwen3-reranker reference server")


@app.on_event("startup")
def _load() -> None:
    # 左 padding：取的是「最后一个位置」的 logit，右 padding 会让批内短样本
    # 的最后一位落在 pad 上，读到的就不是答案位。
    app.state.tok = AutoTokenizer.from_pretrained(_ARGS.model_dir, padding_side="left")
    app.state.model = AutoModelForCausalLM.from_pretrained(
        _ARGS.model_dir, dtype=torch.float16, device_map="cuda",
    ).eval()
    app.state.yes = app.state.tok.convert_tokens_to_ids("yes")
    app.state.no = app.state.tok.convert_tokens_to_ids("no")
    app.state.pre = app.state.tok.encode(_PREFIX, add_special_tokens=False)
    app.state.suf = app.state.tok.encode(_SUFFIX, add_special_tokens=False)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/rerank")
def rerank(req: RerankRequest) -> dict:
    tok, model = app.state.tok, app.state.model
    if not req.documents:
        return {"results": []}

    pairs = [
        f"<Instruct>: {_INSTRUCT}\n<Query>: {req.query}\n<Document>: {doc}"
        for doc in req.documents
    ]
    scores: list[float] = []
    with torch.no_grad():
        for start in range(0, len(pairs), _BATCH):
            chunk = pairs[start : start + _BATCH]
            # padding=False + return_attention_mask=False 是**必须的**，不是风格选择：
            # 下面要给 input_ids 前后拼 prefix/suffix，长度会变。若此处先生成
            # attention_mask，它对应的是拼接前的长度，pad() 拿到长短不一的两者后会
            # 造出全错的 mask。症状极具迷惑性——服务照常返回 200、分数在 (0,1) 区间、
            # 看着完全正常，但**同一请求里所有文档拿到一模一样的分数**，
            # 因为文档内容根本没进模型。排序器于是变成恒等变换，评测指标不动，
            # 很容易被读成"精排没什么用"。
            enc = tok(
                chunk, add_special_tokens=False, truncation=True, padding=False,
                return_attention_mask=False,
                max_length=_MAX_LEN - len(app.state.pre) - len(app.state.suf),
            )
            enc["input_ids"] = [app.state.pre + ids + app.state.suf for ids in enc["input_ids"]]
            enc = tok.pad(enc, padding=True, return_tensors="pt").to(model.device)

            logits = model(**enc).logits[:, -1, :]
            stacked = torch.stack([logits[:, app.state.no], logits[:, app.state.yes]], dim=1)
            probs = torch.nn.functional.log_softmax(stacked.float(), dim=1)[:, 1].exp()
            scores.extend(probs.cpu().tolist())

    return {"results": [{"index": i, "relevance_score": s} for i, s in enumerate(scores)]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-Reranker 参考服务端")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11437)
    _ARGS = parser.parse_args()
    uvicorn.run(app, host=_ARGS.host, port=_ARGS.port, log_level="warning")
