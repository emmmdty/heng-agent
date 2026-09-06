# -*- coding: utf-8 -*-
"""DepositStore —— 结构化沉淀的持久化（#12 任务 B，M0-c）。

存储走 memory 层：与 JsonFilePreferenceStore 同一目录约定（data_dir 下按
buyer 分文件），JSON Lines 逐行一条沉淀，append-only——沉淀是记账，改写
历史条目等于改写对账凭证。

写入门做在 append 里：构造不出确定性验证器的条目在这里被拒（不可验证 =
不许写入，冻结红线），并且**不落任何一行**——半写入的沉淀库比空库更坏，
它让"可验证率 100%"的护栏读数失去分母。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from app.domain.buyer.deposit import MemoryDeposit, build_verifier

# buyer_id 直接拼文件名：只放行安全字符，防路径逃逸（eval 派生的 buyer_id
# 本来可控，但存储层不依赖上游的自觉）
_SAFE_BUYER_ID = re.compile(r"^[A-Za-z0-9._-]+$")


class DepositStore:
    def __init__(self, data_dir: str) -> None:
        self._data_dir = Path(data_dir) / "deposits"

    def _path_for(self, buyer_id: str) -> Path:
        if not _SAFE_BUYER_ID.match(buyer_id):
            raise ValueError(f"buyer_id 含非法字符（只许字母/数字/._-）：{buyer_id!r}")
        return self._data_dir / f"{buyer_id}.jsonl"

    def append(self, deposit: MemoryDeposit) -> None:
        """写入一条沉淀；同 deposit_id 幂等去重。

        verifier_spec 在 MemoryDeposit 构造时已经验证过一次；这里再验一次是
        防御直接从 from_dict/JSON 反序列化进来的条目（绕过构造校验的口子
        不存在——from_dict 也走 __post_init__——但存储层不依赖调用方的自觉）。
        """
        build_verifier(deposit.verifier_spec)
        path = self._path_for(deposit.buyer_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        if any(existing.deposit_id == deposit.deposit_id for existing in self.list_by_buyer(deposit.buyer_id)):
            return
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(deposit.to_dict(), ensure_ascii=False) + "\n")

    def list_by_buyer(self, buyer_id: str) -> list[MemoryDeposit]:
        path = self._path_for(buyer_id)
        if not path.is_file():
            return []
        deposits: list[MemoryDeposit] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            deposits.append(MemoryDeposit.from_dict(json.loads(line)))
        return deposits
