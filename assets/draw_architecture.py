# -*- coding: utf-8 -*-
"""README 架构图生成器（系统 python3 + matplotlib，零仓库依赖）。

    python3 assets/draw_architecture.py            # 产出 light + dark 两张 PNG

    architecture.png        浅色主题
    architecture-dark.png   深色主题（GitHub dark mode 经 <picture> 自动切换）

布局：左侧主列 = 买家交互 → 接入 → 编排（skill 阶段注入）→ 工具 → 基础设施 → 领域，
DDD 洋葱自上而下；右侧竖列 = 可信度工程（评测/门禁/飞轮），两条虚线"会话流水 / 判读反哺"
把评测闭环画进图里——起点终点都钉在盒子边缘，不穿任何元素。
"""
from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# 中文字体：Windows 挂载盘的微软雅黑 / 备选 Noto Sans SC / AR PL UMing
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

# 主题色板：fill / edge / band-title / body-text
THEMES = {
    "light": dict(
        file="architecture.png",
        bg="#ffffff", ink="#1f2328", muted="#57606a", grid="#d0d7de",
        gray=("#f6f8fa", "#8b949e"),
        blue=("#eff6ff", "#2563eb", "#1d4ed8"),
        blue_hi="#dbeafe",
        purple=("#f5f3ff", "#7c3aed", "#6d28d9"),
        green=("#f0fdf4", "#16a34a", "#15803d"),
        red=("#fef2f2", "#dc2626", "#b91c1c"),
        amber=("#fffbf5", "#ea580c", "#c2410c"),
        footer=("#fafbfc", "#d0d7de"),
    ),
    "dark": dict(
        file="architecture-dark.png",
        bg="#0d1117", ink="#e6edf3", muted="#a0a8b4", grid="#30363d",
        gray=("#161b22", "#8b949e"),
        blue=("#12233f", "#4493f8", "#79b8ff"),
        blue_hi="#1f3a5f",
        purple=("#201a38", "#a371f7", "#c9a6ff"),
        green=("#12261c", "#3fb950", "#7ee2a8"),
        red=("#2f1517", "#f85149", "#ff7b72"),
        amber=("#271a0e", "#f0883e", "#ffa657"),
        footer=("#161b22", "#30363d"),
    ),
}


def draw(t):
    W, H = 14.0, 8.8
    fig, ax = plt.subplots(figsize=(W, H), dpi=200)
    fig.patch.set_facecolor(t["bg"])
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

    def text(x, y, s, size=10.5, color=None, weight="normal", ha="center", va="center", rotation=0):
        ax.text(x, y, s, fontsize=size, color=color or t["ink"], fontweight=weight,
                ha=ha, va=va, zorder=3, linespacing=1.45, rotation=rotation)

    def arrow(x1, y1, x2, y2, color=None, lw=1.6, style="-|>", dashed=False, conn="arc3,rad=0"):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
            color=color or t["muted"], linewidth=lw, linestyle=(0, (4, 2)) if dashed else "solid",
            connectionstyle=conn, zorder=4, shrinkA=2, shrinkB=2,
        ))

    def band_label(x, y, s):
        ax.text(x, y, s, fontsize=9, color=t["muted"], ha="center", va="center",
                zorder=3, rotation=0)

    # ============ 左侧主列 ============
    MAIN_L, MAIN_R = 4, 66
    BAND_L = MAIN_L + 1.5
    BAND_R = BAND_L + 60          # 各层带右缘 = 65.5

    # —— 买家交互 ——
    box(8, 88, 22, 8, *t["gray"])
    text(19, 93.2, "React 前端", 11.5, weight="bold")
    text(19, 90.2, "对话流 · 商品卡 · 事件时间线", 8.5, t["muted"])
    box(36, 88, 26, 8, *t["gray"])
    text(49, 93.2, "FastAPI 服务", 11.5, weight="bold")
    text(49, 90.2, "POST /commerce/intents · /events WebSocket", 8.5, t["muted"])
    box(8, 78, 54, 6.4, t["bg"], t["grid"], lw=1.0, dashed=True)
    text(35, 81.2, "Redis Stream 队列削峰（可选）· 幂等键防重复下单 · 独立 Worker 消费进程", 8.8, t["muted"])
    arrow(30, 92, 36, 92)
    arrow(49, 88, 49, 84.4, dashed=True)
    arrow(22, 88, 22, 84.4, dashed=True)
    band_label(2.8, 92, "买家\n交互")
    band_label(2.8, 81, "接入")

    # —— 编排层 ——
    band_fill, band_edge, band_title = t["blue"]
    box(BAND_L, 63.5, 60, 12.2, band_fill, band_edge, lw=1.6)
    text(BAND_L + 1.2, 74.2, "编排层（application/agents）", 8.5, band_title, ha="left")
    text(BAND_L + 30, 64.35, "满足可并行 / 上下文隔离 / 链深才派发，否则 MainAgent 单干",
         7.2, t["muted"], ha="center")
    box(BAND_L + 2.8, 66.4, 16.6, 7.0, t["bg"], band_edge, lw=1.3)
    text(BAND_L + 11.1, 71.6, "Orchestrator", 10.5, weight="bold")
    text(BAND_L + 11.1, 68.7, "会话恢复 · 记忆注入\n技能阶段路由 set_stage", 7.6, t["muted"])
    box(BAND_L + 21.5, 66.4, 17.5, 7.0, t["blue_hi"], band_edge, lw=1.8)
    text(BAND_L + 30.2, 71.8, "MainAgent", 11, weight="bold")
    text(BAND_L + 30.2, 68.6, "skill 阶段拼装 system prompt\n持有全部业务工具，单干优先", 7.6, t["muted"])
    box(BAND_L + 41.5, 68.0, 16, 5.4, t["bg"], band_edge, lw=1.2)
    text(BAND_L + 49.5, 71.7, "SearchAgent", 9.5, weight="bold")
    text(BAND_L + 49.5, 69.4, "检索专家", 7.6, t["muted"])
    box(BAND_L + 41.5, 61.6, 16, 5.4, t["bg"], band_edge, lw=1.2)
    text(BAND_L + 49.5, 65.3, "TradeAgent", 9.5, weight="bold")
    text(BAND_L + 49.5, 63.0, "交易专家", 7.6, t["muted"])
    arrow(BAND_L + 19.4, 69.9, BAND_L + 21.5, 69.9)
    arrow(BAND_L + 39.0, 70.5, BAND_L + 41.5, 70.5, conn="arc3,rad=-0.10")
    arrow(BAND_L + 39.0, 68.8, BAND_L + 41.5, 64.2, conn="arc3,rad=0.10")
    text(BAND_L + 40.2, 74.6, "task_dispatch", 7.0, t["muted"], ha="center")
    band_label(2.8, 69, "编排\n应用层")

    # —— 工具层 ——
    band_fill, band_edge, band_title = t["purple"]
    box(BAND_L, 48.5, 60, 9.2, band_fill, band_edge, lw=1.4)
    text(BAND_L + 1.2, 56.2, "工具层（application/tools）——返回值自带边界与出处", 8.5, band_title, ha="left")
    tools = ["商品检索\n混合召回", "组合报价\n一次履约计费", "组合优化\n预算枚举",
             "订单三件套\n跨越确认", "偏好记忆\n跨会话", "品类洞察\nRAG"]
    tw = 9.0
    for i, s in enumerate(tools):
        tx = BAND_L + 1.6 + i * (tw + 0.62)
        box(tx, 49.6, tw, 5.6, t["bg"], band_edge, lw=1.0, radius=0.8)
        text(tx + tw / 2, 52.4, s, 7.6)
    band_label(2.8, 53, "工具")

    # —— 基础设施层 ——
    band_fill, band_edge, band_title = t["green"]
    box(BAND_L, 29.6, 60, 16.4, band_fill, band_edge, lw=1.4)
    text(BAND_L + 1.2, 44.6, "基础设施（infrastructure）", 8.5, band_title, ha="left")

    infra = [
        ("LLM 网关治理", "限并发 + 间隔闸门\n瞬时故障退避重试\n回退备用模型·发事件\n重试收口只有一层"),
        ("检索链路", "BM25 + 向量 RRF 融合\ncross-encoder 精排\n五档降级链如实标注\n置信度门控（降级态）"),
        ("存储与缓存", "SQLite 流水/订单/会话\nQdrant 向量与知识库\n语义缓存（写旁路）\nRedis 事件背板"),
    ]
    for i, (title, detail) in enumerate(infra):
        ix = BAND_L + 1.6 + i * 19.0
        iw = 17.4 if i < 2 else 18.8
        box(ix, 31.6, iw, 11.6, t["bg"], band_edge, lw=1.2)
        text(ix + iw / 2, 42.3, title, 9.5, weight="bold")
        text(ix + iw / 2, 39.9, detail, 7.2, t["muted"], va="top")
    band_label(2.8, 38, "基础\n设施")

    arrow(BAND_L + 30.2, 63.5, BAND_L + 30.2, 57.9)   # 编排 → 工具
    arrow(BAND_L + 30.2, 48.5, BAND_L + 30.2, 46.2)   # 工具 → 基础设施

    # —— 域层横条 ——
    band_fill, band_edge, band_title = t["red"]
    box(BAND_L, 20.4, 60, 6.6, band_fill, band_edge, lw=1.4)
    text(BAND_L + 30, 24.8, "领域层（domain）—— 计价 / 关税 / 汇率 / 订单状态机：确定性的都留在这里",
         9.3, band_title, weight="bold")
    text(BAND_L + 30, 22.0, "到手价与组合总价可确定性复算 · 免税额度存原生口径 · 订单归属校验",
         7.8, t["muted"])
    arrow(BAND_L + 30, 29.6, BAND_L + 30, 27.2)
    band_label(2.8, 23.6, "领域")

    # ============ 右侧：可信度工程 ============
    EV_L, EV_R = 70.5, 96.5
    ev_fill, ev_edge, ev_title = t["amber"]
    box(EV_L, 21.5, EV_R - EV_L, 74.5, ev_fill, ev_edge, lw=1.4)
    text((EV_L + EV_R) / 2, 92.8, "可信度工程（eval/）", 12.5, ev_title, weight="bold")
    text((EV_L + EV_R) / 2, 89.6, "每个数字都有出处、判据、复现命令", 8.3, t["muted"])

    ev_boxes = [
        ("Rubric 评测回归", "60 用例（44 主线 + 11 红队 + 5 记忆）× judge\n配置行自报指纹与依赖实测可达性"),
        ("A/B 认证轮（mem_replay）", "注入开 vs 关 · 成对比较\n位置互换 + 多数投票 + 双 judge\n三重认证 · 分层判读（预登记）"),
        ("确定性判据门禁（零 LLM）", "金额出处 / 算式自洽 / 组合错加\n知识库出处 / 订单归属 / 召回指标\nmake check 八项 · 审计十几秒 · 真红留档"),
        ("Bad-case 飞轮", "失败采集 → 指纹去重\n人工分诊 → 回归集\n判据与提示词同步迭代"),
    ]
    ey = 84.0
    heights = [12.6, 12.6, 12.6, 9.8]
    gap = 2.4
    for (title, detail), bh in zip(ev_boxes, heights):
        box(EV_L + 2, ey - bh, EV_R - EV_L - 4, bh, t["bg"], ev_edge, lw=1.0)
        text((EV_L + EV_R) / 2, ey - 1.7, title, 10, ev_title, weight="bold")
        text((EV_L + EV_R) / 2, ey - bh / 2 - 1.9, detail, 7.8, t["muted"])
        ey -= bh + gap

    # 闭环两条虚线：起点/终点都钉在盒子边缘，直线连接，不穿任何元素
    # 会话流水：编排层右缘 (65.5, 69.5) → 可信度左缘 (70.5, 78.2)
    arrow(BAND_R, 69.5, EV_L, 78.2, color=ev_edge, dashed=True, lw=1.5)
    text(68.0, 74.6, "会话流水", 7.2, ev_title, rotation=90)
    # 判读反哺：可信度左缘 (70.5, 40.0) → 工具层右缘 (65.5, 52.0)
    arrow(EV_L, 40.0, BAND_R, 52.0, color=ev_edge, dashed=True, lw=1.5)
    text(68.0, 45.4, "判读反哺", 7.2, ev_title, rotation=90)

    # ============ 标题与底注 ============
    text(50, 97.8, "「衡 · Heng」跨境电商 Agent —— 系统架构与可信度工程", 15.5, t["ink"], weight="bold")

    f_fill, f_edge = t["footer"]
    box(4, 8.6, 92.5, 8.8, f_fill, f_edge, lw=1.0)
    text(5.6, 15.4, "认证读数", 8.8, ev_title, weight="bold", ha="left")
    text(13.4, 15.4, "记忆层认证 #12：decisive 33 · p=0.000324（注入开显著优，三重认证全过）　｜　"
                     "Skill 加载 #14：prompt P50 -37.1% · 五护栏全过（42/44 · 0.9545 · 调用率 +12.9%）",
         8.5, t["ink"], ha="left")
    text(5.6, 12.6, "技术栈", 8.8, t["muted"], weight="bold", ha="left")
    text(13.4, 12.6, "Python 3.11 · AgentScope 2.0 · FastAPI · React 18 · SQLite / Qdrant / Redis · 自建 Qwen3 检索服务"
                     "　　边界声明：自建评测 · 无真实线上流量 · 单人标注（方法验证项目）",
         8.2, t["muted"], ha="left")

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--theme", choices=("light", "dark", "both"), default="both")
    args = parser.parse_args()
    names = ("light", "dark") if args.theme == "both" else (args.theme,)
    here = Path(__file__).resolve().parent
    for name in names:
        fig = draw(THEMES[name])
        out = here / THEMES[name]["file"]
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=THEMES[name]["bg"])
        print(f"saved: {out} (theme={name}, font={FONT})")
        plt.close(fig)
