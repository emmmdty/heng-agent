# -*- coding: utf-8 -*-
"""README 架构图生成器（系统 python3 + matplotlib，零仓库依赖）。

    python3 assets/draw_architecture.py            # 产出 light + dark 两张 PNG

    architecture.png        浅色主题
    architecture-dark.png   深色主题（GitHub dark mode 经 <picture> 自动切换）

布局纪律（视觉评审后定稿）：
- 左列各层带 y 间隙统一 2.0-2.2，层间箭头等长；
- 编排层容器完整包住 SearchAgent/TradeAgent，说明文字上移到标题行右侧，不压任何盒子；
- 右列四盒等距（gap 3.0），盒内"标题在上、正文 va=top"，不再上松下紧；
- 盒高按正文行数配：3 行给 13.4、2 行给 10.6，别让盒子空出半截；
- 接入层只接 FastAPI 一条入队箭头（浏览器不直连队列）；
- 两条橙色虚线起止钉在盒边缘，走两列之间的空白通道，竖排标签带底色遮罩压线；
- 底部信息条三行（认证读数 ×2 + 技术栈/边界），每行右端距边框 ≥ 10 单位。
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

# 主题色板：fill / edge / band-title
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

BAND_L = 5.5            # 左列各层带左缘
BAND_W = 60.0
BAND_R = BAND_L + BAND_W   # 65.5
EV_L, EV_R = 70.5, 96.5    # 右列


def draw(t):
    fig, ax = plt.subplots(figsize=(14.0, 8.8), dpi=200)
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

    def text(x, y, s, size=10.5, color=None, weight="normal", ha="center", va="center",
             rotation=0, on_line=False):
        kw = {}
        if on_line:  # 竖排标签压线时加底色遮罩
            kw["bbox"] = dict(boxstyle="round,pad=0.18", facecolor=t["bg"], edgecolor="none")
        ax.text(x, y, s, fontsize=size, color=color or t["ink"], fontweight=weight,
                ha=ha, va=va, zorder=3, linespacing=1.5, rotation=rotation, **kw)

    def arrow(x1, y1, x2, y2, color=None, lw=1.6, style="-|>", dashed=False, conn="arc3,rad=0"):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
            color=color or t["muted"], linewidth=lw, linestyle=(0, (4, 2)) if dashed else "solid",
            connectionstyle=conn, zorder=4, shrinkA=0, shrinkB=0,
        ))

    def band_label(x, y, s):
        ax.text(x, y, s, fontsize=9, color=t["muted"], ha="center", va="center",
                zorder=3, rotation=0)

    # ============ 标题 ============
    text(50, 97.6, "「衡 · Heng」跨境电商 Agent —— 系统架构与可信度工程", 15.5, weight="bold")

    # ============ 买家交互 ============
    box(8, 88, 22, 8, *t["gray"])
    text(19, 93.2, "React 前端", 11.5, weight="bold")
    text(19, 90.2, "对话流 · 商品卡 · 事件时间线", 8.5, t["muted"])
    box(36, 88, 26, 8, *t["gray"])
    text(49, 93.2, "FastAPI 服务", 11.5, weight="bold")
    text(49, 90.2, "POST /commerce/intents · /events WebSocket", 8.5, t["muted"])
    arrow(30, 92, 36, 92)
    band_label(2.8, 92, "买家\n交互")

    # ============ 接入（可选异步链） ============
    box(8, 78, 54, 6.4, t["bg"], t["grid"], lw=1.0, dashed=True)
    text(35, 81.2, "Redis Stream 队列削峰（可选）· 幂等键防重复下单 · 独立 Worker 消费进程", 8.8, t["muted"])
    # 入队的只有 FastAPI：浏览器不直连 Redis，React 那条向下箭头是错的（已删）
    arrow(49, 88, 49, 84.4, dashed=True)
    arrow(35.5, 78, 35.5, 76.0, dashed=True)  # 接入 → 编排（补上层间流向）
    band_label(2.8, 81, "接入")

    # ============ 编排层 ============
    band_fill, band_edge, band_title = t["blue"]
    box(BAND_L, 60.8, BAND_W, 15.2, band_fill, band_edge, lw=1.6)     # 60.8-76.0
    text(BAND_L + 1.2, 74.7, "编排层（application/agents）", 8.4, band_title, ha="left")
    text(BAND_R - 1.2, 74.7, "task_dispatch 按链深与可并行性派发，否则 MainAgent 单干",
         7.0, t["muted"], ha="right")
    box(8.3, 66.4, 16.6, 7.2, t["bg"], band_edge, lw=1.3)
    text(16.6, 71.8, "Orchestrator", 10.5, weight="bold")
    text(16.6, 68.7, "会话恢复 · 记忆注入\n技能阶段路由 set_stage", 7.6, t["muted"])
    box(27.0, 66.4, 17.5, 7.2, t["blue_hi"], band_edge, lw=1.8)
    text(35.75, 72.0, "MainAgent", 11, weight="bold")
    text(35.75, 68.6, "skill 阶段拼装 system prompt\n持有全部业务工具，单干优先", 7.6, t["muted"])
    box(47.0, 68.2, 16, 5.2, t["bg"], band_edge, lw=1.2)
    text(55.0, 71.9, "SearchAgent", 9.5, weight="bold")
    text(55.0, 69.5, "检索专家", 7.6, t["muted"])
    box(47.0, 61.8, 16, 5.2, t["bg"], band_edge, lw=1.2)
    text(55.0, 65.5, "TradeAgent", 9.5, weight="bold")
    text(55.0, 63.1, "交易专家", 7.6, t["muted"])
    arrow(24.9, 70.0, 27.0, 70.0)
    arrow(44.5, 71.2, 47.0, 71.5, conn="arc3,rad=-0.08")
    arrow(44.5, 68.8, 47.0, 64.6, conn="arc3,rad=0.10")
    band_label(2.8, 68, "编排\n应用层")

    arrow(35.5, 60.8, 35.5, 58.8)   # 编排 → 工具

    # ============ 工具层 ============
    band_fill, band_edge, band_title = t["purple"]
    box(BAND_L, 48.6, BAND_W, 10.2, band_fill, band_edge, lw=1.4)     # 48.6-58.8
    text(BAND_L + 1.2, 57.4, "工具层（application/tools）——返回值自带边界与出处", 8.4, band_title, ha="left")
    tools = ["商品检索\n混合召回", "组合报价\n一次履约计费", "组合优化\n预算枚举",
             "订单三件套\n跨越确认", "偏好记忆\n跨会话", "品类洞察\nRAG"]
    for i, s in enumerate(tools):
        tx = 7.1 + i * 9.62
        box(tx, 49.4, 9.0, 6.6, t["bg"], band_edge, lw=1.0, radius=0.8)
        text(tx + 4.5, 52.7, s, 7.6)
    band_label(2.8, 53.5, "工具")

    arrow(35.5, 48.6, 35.5, 46.4)   # 工具 → 基础设施

    # ============ 基础设施层 ============
    band_fill, band_edge, band_title = t["green"]
    box(BAND_L, 29.4, BAND_W, 17.0, band_fill, band_edge, lw=1.4)     # 29.4-46.4
    text(BAND_L + 1.2, 45.0, "基础设施（infrastructure）", 8.4, band_title, ha="left")
    infra = [
        ("LLM 网关治理", "限并发 + 间隔闸门\n瞬时故障退避重试\n回退备用模型·发事件\n重试收口只有一层", 7.1, 17.6),
        ("检索链路", "BM25 + 向量 RRF 融合\ncross-encoder 精排\n降级链如实标注 recall_strategy\n置信度门控（降级态）", 26.3, 17.6),
        ("存储与缓存", "SQLite 流水/订单/会话\nQdrant 向量与知识库\n语义缓存（写旁路）\nRedis 事件背板", 45.5, 18.4),
    ]
    for title, detail, ix, iw in infra:
        box(ix, 30.6, iw, 12.8, t["bg"], band_edge, lw=1.2)           # 30.6-43.4
        text(ix + iw / 2, 41.8, title, 9.5, weight="bold")
        text(ix + iw / 2, 39.8, detail, 7.2, t["muted"], va="top")
    band_label(2.8, 38, "基础\n设施")

    arrow(35.5, 29.4, 35.5, 27.2)   # 基础设施 → 领域

    # ============ 领域层 ============
    band_fill, band_edge, band_title = t["red"]
    box(BAND_L, 20.6, BAND_W, 6.6, band_fill, band_edge, lw=1.4)      # 20.6-27.2
    text(BAND_L + 30, 24.9, "领域层（domain）—— 计价 / 关税 / 汇率 / 订单状态机：确定性的都留在这里",
         9.3, band_title, weight="bold")
    text(BAND_L + 30, 22.1, "到手价与组合总价可确定性复算 · 免税额度存原生口径 · 订单归属校验", 7.8, t["muted"])
    band_label(2.8, 23.9, "领域")

    # ============ 右侧：可信度工程 ============
    ev_fill, ev_edge, ev_title = t["amber"]
    box(EV_L, 24.0, EV_R - EV_L, 72.0, ev_fill, ev_edge, lw=1.4)      # 24.0-96.0
    text((EV_L + EV_R) / 2, 93.6, "可信度工程（eval/）", 12.5, ev_title, weight="bold")
    text((EV_L + EV_R) / 2, 90.3, "每个数字都有出处、判据、复现命令", 8.3, t["muted"])

    ev_boxes = [
        ("Rubric 评测回归",
         "60 用例（44 主线 + 11 红队 + 5 记忆）× judge\njudge 分级 P0/P1/P2 · 主线 44 条固定考卷\n配置行自报指纹与依赖实测可达性", 13.4),
        ("A/B 认证轮（mem_replay）",
         "注入开 vs 关 · 成对比较\n位置互换 + 多数投票 + 双 judge\n三重认证 · 分层判读（预登记）", 13.4),
        ("确定性判据门禁（零 LLM）",
         "金额出处 / 算式自洽 / 组合错加\n知识库出处 / 订单归属 / 召回指标\nmake check 八项 · 审计十几秒 · 真红留档", 13.4),
        ("Bad-case 飞轮",
         "失败采集 → 指纹去重\n人工分诊 → 回归集\n判据与提示词同步迭代", 10.6),
    ]
    ey = 86.6
    for title, detail, bh in ev_boxes:
        box(EV_L + 1.4, ey - bh, EV_R - EV_L - 2.8, bh, t["bg"], ev_edge, lw=1.0)
        text((EV_L + EV_R) / 2, ey - 1.8, title, 10, ev_title, weight="bold")
        text((EV_L + EV_R) / 2, ey - 3.6, detail, 7.8, t["muted"], va="top")
        ey -= bh + 3.0

    # 闭环两条虚线：起止钉在盒边缘，走两列间空白通道
    arrow(BAND_R, 69.5, EV_L, 78.5, color=ev_edge, dashed=True, lw=1.5)
    text(68.0, 74.8, "会话流水", 8.6, ev_title, rotation=90, on_line=True)
    arrow(EV_L, 44.0, BAND_R, 52.0, color=ev_edge, dashed=True, lw=1.5)
    text(68.0, 47.6, "判读反哺", 8.6, ev_title, rotation=90, on_line=True)

    # ============ 底部信息条 ============
    f_fill, f_edge = t["footer"]
    box(4, 7.4, 92.5, 10.2, f_fill, f_edge, lw=1.0)
    text(5.8, 15.3, "认证读数", 8.8, ev_title, weight="bold", ha="left")
    text(13.6, 15.3, "记忆层认证 #12：decisive 33 · p=0.000324（注入开显著优，三重认证全过）",
         8.6, t["ink"], ha="left")
    text(13.6, 12.9, "Skill 渐进加载 #14：prompt P50 -37.1% · 五护栏全过（42/44 · 均分 0.9545 · 调用率 +12.9%）",
         8.6, t["ink"], ha="left")
    text(5.8, 10.0, "技术栈", 8.8, t["muted"], weight="bold", ha="left")
    text(13.6, 10.0, "Python 3.11 · AgentScope 2.0 · FastAPI · React 18 · SQLite / Qdrant / Redis · 自建 Qwen3 检索"
                     "　|　边界声明：自建评测 · 无真实线上流量 · 单人标注（方法验证）",
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
