# -*- coding: utf-8 -*-
"""README 架构图生成器（零依赖仓库：用系统 python3 + matplotlib 跑）。

    python3 docs/assets/draw_architecture.py   # 产出 docs/assets/architecture.png

布局：左侧主列 = 买家交互 → 接入 → 编排（skill 阶段注入）→ 工具 → 基础设施，
DDD 洋葱自上而下；右侧竖列 = 可信度工程（评测/门禁/飞轮），虚线反馈箭头
回到主列——评测闭环是这个项目的一等公民，值得画进去。
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# 中文字体：Windows 挂载盘的微软雅黑 / 备选 Noto Sans SC
for candidate in (
    "/mnt/c/Windows/Fonts/msyh.ttc",
    "/mnt/c/Windows/Fonts/NotoSansSC-VF.ttf",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
):
    if Path(candidate).exists():
        fm.fontManager.addfont(candidate)
        FONT = fm.FontProperties(fname=candidate).get_name()
        break
else:
    FONT = "sans-serif"

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = FONT
plt.rcParams["axes.unicode_minus"] = False

BG = "#ffffff"
INK = "#1f2328"
MUTED = "#57606a"
GRID = "#d0d7de"

BLUE = ("#dbeafe", "#2563eb")      # 编排/Agent
PURPLE = ("#ede9fe", "#7c3aed")    # 工具
GREEN = ("#dcfce7", "#16a34a")     # 基础设施
GRAY = ("#f6f8fa", "#8b949e")      # 接入
AMBER = ("#fff7ed", "#ea580c")     # 可信度工程（差异化重点）
RED = ("#fee2e2", "#dc2626")       # 域内硬约束

W, H = 14.0, 8.6
fig, ax = plt.subplots(figsize=(W, H), dpi=200)
fig.patch.set_facecolor(BG)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def box(x, y, w, h, fill, edge, lw=1.4, dashed=False, radius=1.2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fill, edgecolor=edge, linewidth=lw,
        linestyle=(0, (4, 2)) if dashed else "solid",
        zorder=2,
    ))


def text(x, y, s, size=10.5, color=INK, weight="normal", ha="center", va="center", rotation=0):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, zorder=3, linespacing=1.45, rotation=rotation)


def arrow(x1, y1, x2, y2, color=MUTED, lw=1.6, style="-|>", dashed=False, conn="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
        color=color, linewidth=lw, linestyle=(0, (4, 2)) if dashed else "solid",
        connectionstyle=conn, zorder=4, shrinkA=2, shrinkB=2,
    ))


def band_label(x, y, s):
    ax.text(x, y, s, fontsize=9, color=MUTED, ha="center", va="center",
            zorder=3, rotation=0)


# ============ 左侧主列 ============
MAIN_L, MAIN_R = 4, 66
BAND_L = MAIN_L + 1.5

# —— 买家交互 ——
box(8, 88, 22, 8, *GRAY)
text(19, 93.2, "React 前端", 11.5, weight="bold")
text(19, 90.2, "对话流 · 商品卡 · 事件时间线", 8.5, MUTED)
box(36, 88, 26, 8, *GRAY)
text(49, 93.2, "FastAPI 服务", 11.5, weight="bold")
text(49, 90.2, "POST /commerce/intents · /events WebSocket", 8.5, MUTED)
box(8, 78, 54, 6.4, "#fbfbfc", GRID, lw=1.0, dashed=True)
text(35, 81.2, "Redis Stream 队列削峰（可选）· 幂等键防重复下单 · 独立 Worker 消费进程", 8.8, MUTED)
arrow(30, 92, 36, 92)
arrow(49, 88, 49, 84.4, dashed=True)
arrow(22, 88, 22, 84.4, dashed=True)
band_label(2.8, 92, "买家\n交互")
band_label(2.8, 81, "接入")

# —— 编排层 ——
box(BAND_L, 63.5, 60, 12.2, "#eff6ff", "#2563eb", lw=1.8)
text(BAND_L + 1.2, 74.2, "编排层（application/agents）", 8.5, "#2563eb", ha="left")
text(BAND_L + 21, 64.35, "满足可并行 / 上下文隔离 / 链深才派发，否则 MainAgent 单干", 6.6, MUTED, ha="center")
box(BAND_L + 2.8, 65.6, 16.6, 7.6, "#ffffff", "#2563eb", lw=1.4)
text(BAND_L + 11.1, 71.2, "Orchestrator", 10.5, weight="bold")
text(BAND_L + 11.1, 68.1, "会话恢复 · 记忆注入\n技能阶段路由 set_stage", 7.8, MUTED)
box(BAND_L + 21.5, 65.6, 17.5, 7.6, "#dbeafe", "#2563eb", lw=2.0)
text(BAND_L + 30.2, 71.4, "MainAgent", 11, weight="bold")
text(BAND_L + 30.2, 67.9, "skill 阶段拼装 system prompt\n持有全部业务工具，单干优先", 7.8, MUTED)
box(BAND_L + 41.5, 67.4, 16, 5.8, "#ffffff", "#2563eb")
text(BAND_L + 49.5, 71.2, "SearchAgent", 9.5, weight="bold")
text(BAND_L + 49.5, 68.9, "检索专家", 7.8, MUTED)
box(BAND_L + 41.5, 60.2, 16, 5.8, "#ffffff", "#2563eb")
text(BAND_L + 49.5, 64, "TradeAgent", 9.5, weight="bold")
text(BAND_L + 49.5, 61.7, "交易专家", 7.8, MUTED)
arrow(BAND_L + 19.4, 69.4, BAND_L + 21.5, 69.4)
arrow(BAND_L + 39, 70.2, BAND_L + 41.5, 70.2, conn="arc3,rad=-0.12")
arrow(BAND_L + 39, 68.4, BAND_L + 41.5, 63.4, conn="arc3,rad=0.12")
text(BAND_L + 40.3, 74.6, "task_dispatch", 7.0, MUTED, ha="center")
band_label(2.8, 69, "编排\n应用层")

# —— 工具层 ——
box(BAND_L, 48.5, 60, 9.2, "#f5f3ff", "#7c3aed", lw=1.5)
text(BAND_L + 1.2, 56.2, "工具层（application/tools）——返回值自带边界与出处", 8.5, "#7c3aed", ha="left")
tools = ["商品检索\n混合召回", "组合报价\n一次履约计费", "组合优化\n预算枚举", "订单三件套\n跨越确认", "偏好记忆\n跨会话", "品类洞察\nRAG"]
tw = 9.0
for i, t in enumerate(tools):
    tx = BAND_L + 1.6 + i * (tw + 0.62)
    box(tx, 49.6, tw, 5.6, "#ffffff", "#7c3aed", lw=1.0, radius=0.8)
    text(tx + tw / 2, 52.4, t, 7.6)
band_label(2.8, 53, "工具")

# —— 基础设施层 ——
box(BAND_L, 29.6, 60, 16.4, "#f0fdf4", "#16a34a", lw=1.5)
text(BAND_L + 1.2, 44.6, "基础设施（infrastructure）", 8.5, "#16a34a", ha="left")

box(BAND_L + 1.6, 31.6, 17, 11.6, "#ffffff", "#16a34a")
text(BAND_L + 10, 42.3, "LLM 网关治理", 9.5, weight="bold")
text(BAND_L + 10, 40.0, "限并发 + 间隔闸门\n瞬时故障退避重试\n回退备用模型·发事件\n重试收口只有一层", 7.2, MUTED, va="top")

box(BAND_L + 20.6, 31.6, 17, 11.6, "#ffffff", "#16a34a")
text(BAND_L + 29, 42.3, "检索链路", 9.5, weight="bold")
text(BAND_L + 29, 40.0, "BM25 + 向量 RRF 融合\ncross-encoder 精排\n五档降级链如实标注\n置信度门控（降级态）", 7.2, MUTED, va="top")

box(BAND_L + 39.6, 31.6, 18.8, 11.6, "#ffffff", "#16a34a")
text(BAND_L + 49, 42.3, "存储与缓存", 9.5, weight="bold")
text(BAND_L + 49, 40.0, "SQLite 流水/订单/会话\nQdrant 向量与知识库\n语义缓存（写旁路）\nRedis 事件背板", 7.2, MUTED, va="top")
band_label(2.8, 38, "基础\n设施")

arrow(BAND_L + 30.2, 63.5, BAND_L + 30.2, 57.7)   # 编排 → 工具
arrow(BAND_L + 30.2, 48.5, BAND_L + 30.2, 46.2)   # 工具 → 基础设施

# —— 域层横条 ——
box(BAND_L, 20.4, 60, 6.6, *RED)
text(BAND_L + 30, 24.8, "领域层（domain）—— 计价 / 关税 / 汇率 / 订单状态机：确定性的都留在这里", 9.3, "#991b1b", weight="bold")
text(BAND_L + 30, 22.0, "到手价与组合总价可确定性复算 · 免税额度存原生口径 · 订单归属校验", 7.8, "#b91c1c")
arrow(BAND_L + 30, 29.6, BAND_L + 30, 27.0)
band_label(2.8, 23.6, "领域")

# ============ 右侧：可信度工程 ============
EV_L, EV_R = 70.5, 96.5
box(EV_L, 21.5, EV_R - EV_L, 74.5, "#fffbf5", "#ea580c", lw=2.0)
text((EV_L + EV_R) / 2, 92.8, "可信度工程（eval/）", 12.5, "#c2410c", weight="bold")
text((EV_L + EV_R) / 2, 89.6, "每个数字都有出处、判据、复现命令", 8.3, MUTED)

ev_boxes = [
    ("Rubric 评测回归", "44 用例 × LLM judge（P0/P1/P2）\n配置行自报指纹与依赖实测\n无出处金额率、算式自洽门禁"),
    ("A/B 认证轮（mem_replay）", "注入开 vs 关 · 成对比较\n位置互换 + 多数投票 + 双 judge\n三重认证 · 分层判读（预登记）"),
    ("确定性判据门禁（零 LLM）", "金额出处 / 算式自洽 / 组合错加\n知识库出处 / 订单归属 / 召回指标\nmake check 八项 15 秒 · 真红留档"),
    ("Bad-case 飞轮", "失败采集 → 指纹去重\n人工分诊 → 回归集\n判据与提示词同步迭代"),
]
ey = 84.0
for title, detail in ev_boxes:
    box(EV_L + 2, ey - 13.2, EV_R - EV_L - 4, 13.2, "#ffffff", "#ea580c", lw=1.2)
    text((EV_L + EV_R) / 2, ey - 1.6, title, 10, "#c2410c", weight="bold")
    text((EV_L + EV_R) / 2, ey - 7.4, detail, 7.8, MUTED)
    ey -= 15.6

arrow(66, 71.5, EV_L, 82.0, style="-|>", color="#ea580c", dashed=True, conn="arc3,rad=-0.10")
text(68.1, 79.0, "会话流水", 7.0, "#ea580c", ha="center", va="center", rotation=90)
arrow(EV_L, 40.5, 66, 54.5, style="-|>", color="#ea580c", dashed=True, conn="arc3,rad=0.15")
text(68.1, 46.5, "判读反哺", 7.0, "#ea580c", ha="center", va="center", rotation=90)

# ============ 标题与底注 ============
text(50, 97.8, "「衡 · Heng」跨境电商 Agent —— 系统架构与可信度工程", 15.5, INK, weight="bold")
text(50, 17.3, "认证读数：记忆层 #12 敏感层 decisive 33 / p=0.000324（B=注入开显著优，三重认证全过）",
     8.8, "#57606a")
text(50, 14.6, "skill 渐进加载 #14：prompt P50 -37.1%，五护栏全过（42/44 · 均分 0.9545 · 调用率 +12.9%）",
     8.8, "#57606a")
text(50, 12.0, "技术栈：Python 3.11 · AgentScope 2.0 · FastAPI · React 18 · SQLite/Qdrant/Redis · 自建 Qwen3 检索服务",
     8.2, MUTED)
text(50, 9.2, "边界声明：自建评测、无真实线上流量、单人标注——定位为 Agent 可信度工程的方法验证",
     8.2, MUTED)

out = Path(__file__).resolve().parent / "architecture.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
print(f"saved: {out} (font={FONT})")
