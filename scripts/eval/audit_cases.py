# -*- coding: utf-8 -*-
"""Rubric 用例集自检：商品指代是否唯一

与 `validate_datasets.py` 同一思路——**基准本身错了，后面所有指标都不可信**，
区别在于它查的是召回标注集，本脚本查的是端到端 Rubric 用例集。

要防的具体问题：`eval/cases.yaml` 是按 10 SPU 的旧商品库写的，那时每个品牌只有
一个商品，所以 query 里写「AeroHush 耳机」是无歧义的。商品库扩到 60 SPU 后
同品牌出现了多个变体（AeroHush Pro / AeroHush Lite），而 rubric 往往还钉着
具体的 `P100X`——Agent 合理地选了另一个变体就会被判 FAIL。

这类失败最难查：被测系统没错，评测夹具过期了（同类问题见踩坑档案第 7 条）。
不能靠"跑一遍看谁挂了"来发现，必须静态审计。

**判据必须和真实的消歧机制对齐**：买家是用整句话消歧的，不是只报品牌名。
只看品牌词的判据会把「LumenGo 露营灯」也报成歧义（LumenGo 匹配 2 个 SPU），
而这句话里的"露营灯"已经把候选收敛到一个了。按那种判据去"修"用例，
只会把 query 改得越来越啰嗦，真歧义反而淹没在误报里——自检脚本一旦开始
产噪声，人就会开始忽略它，等于没有。

用法：
    uv run python scripts/eval/audit_cases.py
退出码：0 = 无歧义；1 = 存在指代歧义（在 query 里补足消歧信息，别放宽 rubric）。
"""
from __future__ import annotations

import asyncio
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.infrastructure.persistence.in_memory_repositories import (  # noqa: E402
    InMemoryProductRepository,
)

_CASES = Path("eval/cases.yaml")
# 品牌词按连续 ASCII 词提取；长度 >= 4 用来滤掉 USD / 65W 这类噪声
_BRAND_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9]{3,}")
_PRODUCT_ID = re.compile(r"P\d{4}")
# 标题里的可辨识片段：连续 CJK 段（品类/形态词）与含数字的规格词（30W、100W）
_CJK_RUN = re.compile(r"[一-鿿]{2,}")
_SPEC_TOKEN = re.compile(r"[A-Za-z0-9]*\d+[A-Za-z0-9]*")


@dataclass(frozen=True)
class Candidate:
    product_id: str
    title: str


@dataclass(frozen=True)
class BrandResolution:
    ambiguous: bool
    winner: Optional[str]        # 被 query 唯一收敛到的 product_id
    scores: dict[str, int]


def _discriminators(title: str) -> set[str]:
    """标题里可用来区分同品牌变体的片段。

    CJK 段再切 2-gram：中文没有空格，"便携露营灯"整段几乎不会原样出现在 query 里，
    但"露营"/"营灯"这样的 2-gram 会。不切的话消歧几乎永远失败。
    """
    tokens: set[str] = set()
    for run in _CJK_RUN.findall(title):
        tokens.add(run)
        tokens.update(run[i : i + 2] for i in range(len(run) - 1))
    tokens.update(spec.lower() for spec in _SPEC_TOKEN.findall(title))
    return tokens


def resolve_brand(query: str, candidates: list[Candidate]) -> BrandResolution:
    """整句 query 能否把同品牌的多个候选收敛到唯一一个。

    打分方式刻意简单：数 query 命中了各候选**独有**的辨识词多少个。
    用"独有"而不是"全部"，是因为共有词（两款都是"耳机"）本来就没有区分力，
    把它算进去会让并列的两个候选一起涨分、看起来仍是平手——结论虽然对，
    但换个共有词多的品类就会误判成有赢家。
    """
    if len(candidates) <= 1:
        return BrandResolution(False, candidates[0].product_id if candidates else None, {})

    per_candidate = {item.product_id: _discriminators(item.title) for item in candidates}
    shared: set[str] = set.intersection(*per_candidate.values()) if per_candidate else set()
    lowered = query.lower()
    scores = {
        product_id: sum(1 for token in tokens - shared if token in lowered)
        for product_id, tokens in per_candidate.items()
    }
    best = max(scores.values())
    winners = [pid for pid, score in scores.items() if score == best]
    if best == 0 or len(winners) > 1:
        return BrandResolution(True, None, scores)
    return BrandResolution(False, winners[0], scores)


async def main() -> int:
    products = await InMemoryProductRepository().list_all()
    cases = yaml.safe_load(_CASES.read_text(encoding="utf-8"))["cases"]

    problems: list[tuple[str, str, list[Candidate], set, BrandResolution]] = []
    mismatches: list[tuple[str, str, str, set]] = []
    for case in cases:
        query_text = " ".join(case["queries"])
        rubric_text = yaml.safe_dump(case.get("rubric", {}), allow_unicode=True)
        pinned = set(_PRODUCT_ID.findall(rubric_text))

        for brand in sorted(set(_BRAND_TOKEN.findall(query_text))):
            matched = [
                Candidate(p.product_id, p.title) for p in products
                if brand.lower() in p.title.lower() or brand.lower() in p.brand.lower()
            ]
            if len(matched) <= 1:
                continue
            resolution = resolve_brand(query_text, matched)
            if resolution.ambiguous:
                problems.append((case["id"], brand, matched, pinned, resolution))
            elif pinned and resolution.winner not in pinned and pinned & {c.product_id for c in matched}:
                # query 收敛到了 A，rubric 却钉着同品牌的 B —— 两边打架，必然误判
                mismatches.append((case["id"], brand, resolution.winner, pinned))

    print(f"{_CASES}：{len(cases)} 条用例，商品库 {len(products)} 个 SPU\n")
    if not problems and not mismatches:
        print("用例自检通过：每个品牌指代都能由 query 收敛到唯一商品")
        return 0

    for case_id, brand, matched, pinned, resolution in problems:
        print(f"[{case_id}] \"{brand}\" 匹配 {len(matched)} 个 SPU，query 无法收敛：")
        for candidate in matched:
            mark = " ←rubric 钉死" if candidate.product_id in pinned else ""
            print(f"    {candidate.product_id}  {candidate.title}"
                  f"（消歧词命中 {resolution.scores.get(candidate.product_id, 0)}）{mark}")
        if pinned & {c.product_id for c in matched}:
            print("    风险：Agent 选其他变体属合理行为，却会被判 FAIL")
        print()

    for case_id, brand, winner, pinned in mismatches:
        print(f"[{case_id}] \"{brand}\"：query 指向 {winner}，rubric 却钉着 {sorted(pinned)}——两边打架\n")

    print("修法建议：在 query 里补足消歧信息（如带上品类词或型号），"
          "而不是放宽 rubric——放宽会让用例失去检验能力。")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
