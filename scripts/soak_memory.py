# -*- coding: utf-8 -*-
"""soak 内存验证（二十三期清单 7）：数百会话打服务，验证 RSS 在 LRU 上限后持平

用法：
    uv run python scripts/soak_memory.py                        # 300 会话，并发 3
    uv run python scripts/soak_memory.py --sessions 260 --rss-every 5
    uv run python scripts/soak_memory.py --json

背景：十七期给按会话累积的东西加了 LRU 上限（SESSION_CACHE_MAX，默认 200），
但没有长跑环境验证过"上限真的生效"。本工具补上这个欠账：
每会话独立 session_id / buyer_id，用**读路径与闲聊 query** 轮换打服务
（不用下单——写路径会扣库存、写订单，是对用例集事实状态的污染），
定期采样 uvicorn 进程的 RSS，按会话数越过 LRU 上限前后两段分别拟合
每会话 RSS 增长斜率，输出对照与判读。

**判读是观测不是门禁**（硬断言会把一次环境的抖动固化成红灯），
但样本不足不判读——两头都没有足够采样点时，"无法判读"就是正确答案。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "eval"

# 拐点假设：与服务默认配置同源（tests/test_soak_memory.py 钉住这条一致性）。
# 服务若用 SESSION_CACHE_MAX 改过配置，用 --lru-max 传实际值。
LRU_ASSUMED_MAX = 200

# soak 用的读路径/闲聊 query 轮换池。刻意避开写路径（下单/取消）：
# 三百个 soak 会话要是真的下了三百单，商品库库存与订单事实就被搅乱了。
_SOAK_QUERIES = (
    "你好呀，你们平台能买什么？",
    "帮我查一下订单 HNG-SOAK000 的状态。",
    "有什么适合送人的旅行装备推荐吗？",
)


def pick_query(index: int) -> str:
    return _SOAK_QUERIES[index % len(_SOAK_QUERIES)]


def parse_rss_kb(status_text: str) -> int | None:
    """从 /proc/<pid>/status 文本提取 VmRSS（kB）。取不到就返回 None——
    采样缺失是常态（进程刚好重启、字段缺失），不该让整个 soak 报废。"""
    for line in status_text.splitlines():
        if line.startswith("VmRSS:"):
            value = line.split(":", 1)[1].strip().split()[0]
            try:
                return int(value)
            except ValueError:
                return None
    return None


def read_service_rss(base_url: str) -> int | None:
    """找 soak 目标服务进程并读 RSS。

    审查修正（M3）：此前只按 "uvicorn + app.presentation.server" 匹配、
    base_url 参数收了不用——多实例并存时会静默采到另一个进程的 RSS，
    斜率结论失真且无告警。现在从 base_url 解析端口并在命令行里核对
    （`--port N` 或参数含 :N）；找不到目标进程返回 None，由调用方决定
    首采样时是否硬失败。
    """
    port = _port_of(base_url)
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cmdline = (proc / "cmdline").read_bytes().split(b"\0")
            args = [part.decode(errors="ignore") for part in cmdline if part]
        except OSError:
            continue
        if not any(part.endswith("uvicorn") for part in args):
            continue
        if not any("app.presentation.server" in part for part in args):
            continue
        if "uv" in args[:1]:
            continue  # uv run 包装外壳，真进程带着 .venv 全路径
        if not _serves_port(args, port):
            continue
        try:
            return parse_rss_kb((proc / "status").read_text(encoding="utf-8"))
        except OSError:
            continue
    return None


def _port_of(base_url: str) -> str:
    from urllib.parse import urlparse

    parsed = urlparse(base_url)
    if parsed.port:
        return str(parsed.port)
    return "443" if parsed.scheme == "https" else "80"


def _serves_port(args: list[str], port: str) -> bool:
    for index, part in enumerate(args):
        if part == "--port" and index + 1 < len(args) and args[index + 1] == port:
            return True
        if f":{port}" in part and not part.startswith("--"):
            return True
    return False


def slope_per_session(points: list[tuple[int, int]]) -> float:
    """最小二乘斜率（kB / 会话）。样本 <2 个返回 0——斜率是读数，
    不是凑出来参与判读的。"""
    if len(points) < 2:
        return 0.0
    n = len(points)
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    if denominator == 0:
        return 0.0
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in points)
    return numerator / denominator


def plateau_verdict(before_slope: float, after_slope: float) -> str:
    """按前后两段斜率的比值给判读。

    阈值取 20%：LRU 生效时后段只剩背景波动（索引缓存、碎片），
    不应再有显著的单会话增长。前段本身无增长时比值无意义，明确不判读。
    """
    if before_slope <= 1.0:
        return "前段无增长，无法判读"
    if after_slope <= before_slope * 0.2:
        return "持平"
    return "仍在增长"


@dataclass
class SoakPoint:
    sessions_done: int
    rss_kb: int | None
    ts: str


async def run_soak(
    sessions: int,
    concurrency: int,
    base_url: str,
    lru_max: int,
    rss_every: int,
    progress=print,
) -> dict:
    """打 N 个独立会话并采样 RSS。返回可落盘的读数结构。"""
    samples: list[SoakPoint] = []
    errors: list[str] = []
    done = 0
    lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(concurrency)

    def sample(now_done: int) -> None:
        rss = read_service_rss(base_url)
        if rss is None and now_done == 0:
            # 首采样就找不到目标进程：宁可硬失败，不把"采到别人"当读数
            raise SystemExit(
                f"找不到监听 {base_url} 的 uvicorn 进程（按命令行端口核对）。\n"
                f"  先确认服务在跑：curl -m5 --noproxy 127.0.0.1 {base_url}/health",
            )
        samples.append(SoakPoint(
            sessions_done=now_done,
            rss_kb=rss,
            ts=datetime.now().strftime("%H:%M:%S"),
        ))

    sample(0)  # 起点基线（0 会话）
    started = datetime.now()

    async with httpx.AsyncClient(timeout=180) as client:
        async def one(index: int) -> None:
            nonlocal done
            async with semaphore:
                payload = {
                    "shopping_session_id": f"eval-soak-{index}-{uuid.uuid4().hex[:6]}",
                    "buyer_id": f"soak-buyer-{index}",
                    "locale": "zh-CN",
                    "currency": "CNY",
                    "raw_query": pick_query(index),
                }
                try:
                    response = await client.post(f"{base_url}/commerce/intents", json=payload)
                    response.raise_for_status()
                except Exception as err:  # noqa: BLE001 —— 单会话失败不报废整轮，但必须留名
                    errors.append(f"session-{index}: {type(err).__name__}: {err}")
                    return
                async with lock:
                    done += 1
                    if done % rss_every == 0 or done == sessions:
                        sample(done)
                        rss = samples[-1].rss_kb
                        progress(f"  {done}/{sessions} 会话完成，RSS={rss if rss is not None else '（进程未找到）'} kB")

        await asyncio.gather(*(one(i) for i in range(sessions)))

    elapsed = (datetime.now() - started).total_seconds()
    # 分段：拐点前（会话数 < lru_max）与拐点后（>= lru_max）。
    # 每段至少 3 个采样点才拟合斜率，否则该段斜率置 None（不参与判读）。
    before = [(p.sessions_done, p.rss_kb) for p in samples
              if p.rss_kb is not None and p.sessions_done < lru_max]
    after = [(p.sessions_done, p.rss_kb) for p in samples
             if p.rss_kb is not None and p.sessions_done >= lru_max]
    before_slope = slope_per_session(before) if len(before) >= 3 else None
    after_slope = slope_per_session(after) if len(after) >= 3 else None
    if before_slope is None or after_slope is None:
        verdict = "样本不足，无法判读"
    else:
        verdict = plateau_verdict(before_slope, after_slope)
    return {
        "sessions_requested": sessions,
        "sessions_done": done,
        "errors": errors,
        "elapsed_seconds": round(elapsed, 1),
        "lru_max": lru_max,
        "rss_samples": [
            {"sessions": p.sessions_done, "rss_kb": p.rss_kb, "ts": p.ts} for p in samples
        ],
        "before_slope_kb_per_session": round(before_slope, 1) if before_slope is not None else None,
        "after_slope_kb_per_session": round(after_slope, 1) if after_slope is not None else None,
        "verdict": verdict,
    }


def render(result: dict) -> str:
    lines = ["# Soak 内存验证（LRU 拐点观测）", ""]
    lines.append(
        f"会话 {result['sessions_done']}/{result['sessions_requested']} 完成"
        f"｜失败 {len(result['errors'])}｜用时 {result['elapsed_seconds']:.0f}s｜"
        f"LRU 上限假设 {result['lru_max']}",
    )
    lines.append(
        f"拐点前斜率（< {result['lru_max']} 会话）："
        f"{result['before_slope_kb_per_session']} kB/会话｜"
        f"拐点后斜率：{result['after_slope_kb_per_session']} kB/会话",
    )
    lines.append(f"判读：**{result['verdict']}**")
    if result["errors"]:
        lines.append("")
        lines.append("⚠️ 失败会话（前 5 条）：")
        lines.extend(f"- {item}" for item in result["errors"][:5])
    lines.append("")
    lines.append("| 会话数 | RSS kB | 时刻 |")
    lines.append("|---|---|---|")
    for sample in result["rss_samples"]:
        rss = sample["rss_kb"] if sample["rss_kb"] is not None else "（缺失）"
        lines.append(f"| {sample['sessions']} | {rss} | {sample['ts']} |")
    return "\n".join(lines)


async def main_async(args: argparse.Namespace) -> int:
    result = await run_soak(
        sessions=args.sessions,
        concurrency=args.concurrency,
        base_url=args.base_url.rstrip("/"),
        lru_max=args.lru_max,
        rss_every=args.rss_every,
    )
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = EVAL_DIR / f"soak-{stamp}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else render(result))
    print(f"\n（读数已写入 {out_path}）")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=300)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--lru-max", type=int, default=LRU_ASSUMED_MAX)
    parser.add_argument("--rss-every", type=int, default=10, help="每完成 N 个会话采样一次 RSS")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
