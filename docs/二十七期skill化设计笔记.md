# C1 设计笔记：prompt token 构成与 skill 化候选（#14）

> 2026-09-06，二十七期 C1 产出。数据来源：`eval/skill-breakdown-20260906-183405.json/.md`
>（脚本 `scripts/eval/skill_token_breakdown.py`，零 LLM，只读钉死的 R8 基线流水
> `eval/report-20260905-142017.json` 的 43 会话 / 64 agent 轮）。
> 这是 C2（skill 单元与 loader）的设计依据——没有构成数据不写 loader。

## 一、构成数据（地板锚定口径）

| 构成 | 读数 | 口径 |
|---|---|---|
| prompt token P50 / P95 | **17,083 / 33,863** | 与 `make cost --report` 同一选取与 usage 字段（P50 差 7 来自偶数取中位的实现差异） |
| **固定段真值** | **7,094 token = P50 的 42%** | 零工具、单轮 agent 轮的 prompt_tokens 最小值（chitchat-boundary）——system + 全量工具 schema + 一句问句的直接测量，不做任何折算假设 |
| 固定段字符构成 | system **3,666** chars（heng.yml 单块）＋ 工具 schema **14,578** chars（14 个） | 真 Toolkit 探针逐个 schema 计字符 |
| **历史段** | **P50 的 58%**（prompt − 固定段） | 含对话 prose、**工具返回载荷**（商品卡 JSON 进上下文后随轮累积）、注入块 |
| 历史段极差 | 无工具轮 7,094 ↔ 单轮重检索轮 **50,880**（no-fabrication） | 差距几乎全在工具载荷——单轮检索的 product card JSON 比对话本身贵一个量级 |
| 注入块 | memory-recall ≈144 chars / memory-forget ≈143 chars（估算，流水不记合并 UserMsg） | 占比可忽略；skill 化不动它 |

**结论一：P50 的 42% 是"不管问什么都全量发"的固定段——skill 化的主战场在工具
schema（14,578 chars 里 57% 是零调用工具，见下）与 system prompt 单块全量注入。**

## 二、skill 化候选清单（按确定性证据排序）

### 候选 1：agentscope 内置 Task* 四件套——零调用的纯死重
`TaskUpdate` 3,596 + `TaskCreate` 2,535 + `TaskList` 1,193 + `TaskGet` 1,024
= **8,348 chars = 工具 schema 总量的 57%**。
调用证据：R8 全 44 条 **0 次**；全史流水（1,386 份会话、4,064 次工具事件）
**0 次**。这是框架默认注册的规划工具，本 Agent 从未用过。
处置（C3）：toolkit 注册处排除（agentscope `Toolkit` 默认注入，需查构造参数）。
风险与护栏：删 schema 是行为面变化 → C4 工具调用率不降 + full PASS 不回退一票否决。

### 候选 2：task_dispatch——低频但非死代码
591 chars；R8 主线 0 次、全史 4 次（检索子代理派发）。
处置：**不删**，归入"检索阶段"skill 单元（派发是检索路径的一部分）。
C1 记录它是低频工具——渐进加载天然受益（非检索轮不发它的 schema）。

### 候选 3：system prompt 单块全量注入（3,666 chars，zh 密度 ≈ 1 token/char）
heng.yml 的 main_agent system prompt 是**单块**，确认卡纪律 / 算式口径 /
免税规则等段落在纯检索轮（chitchat、memory-write 7,174 token 的地板区）
也全量发。处置（C2/C3）：按任务阶段拆片段打包进 skill
（检索阶段：检索纪律+预算约束；交易阶段：确认卡+算式+免税；记忆阶段：
能力边界声明），loader 按阶段渐进注入。
约束（沿任务书）：**降级链与字面门控行为不变**——heng.yml 是
prompt_fingerprint 的输入，拆块必改指纹 → C4 前要按"新指纹下重取基线"
的前置纪律走（交接文档开工前置 P0-1 同族）。

### 候选 4（相邻发现，不在 #14 范围）：工具返回载荷是 P95 的主驱动
P95 33,863 vs P50 17,083 的差距几乎全是历史里的工具载荷（商品卡/报价单
JSON 全文进上下文）。#14 的 Skill 范围（提示词片段+工具子集+判据）不覆盖
"工具返回瘦身"（那是 context_policy / 卡片裁剪的事）。**记录备查，不在本
任务书里做**——防止 C4 读数把两项改动的效应混在一起没法归因。

## 三、C2 的直接设计输入

1. **skill 单元的"工具子集"按调用证据分三档**：
   - 常驻（几乎每轮都用）：`product_search_tool`（R8 109 次）
   - 检索阶段：`category_insight_tool`（14）/ `task_dispatch`（0，路径保留）
   - 交易阶段：`quote_basket_tool`（14）/ `optimize_basket_tool`（8）/
     `create_order_tool`（7）/ `cancel_order_tool`（2）/ `query_order_tool`（2）
   - 记忆阶段：`remember_preference_tool`（6）/ `forget_preference_tool`（2）
   - 移除：Task* 四件套（0）
2. **阶段判定**：C2 loader 需要意图→阶段的路由。R8 流水里同一会话会跨阶段
   （long-chain-add-then-switch 等），路由必须**保守**（宁可多带：交易片段
   在纯检索轮多发，也不能检索轮发少了导致行为回退——C4 的 PASS 不回退是
   一票否决）。
3. **权限接线**：新工具集必须过交接文档「五之二」判据（挂 Harness、进白名单、
   schema 断言）；`allow_business_tools` 的 permissions 语义在拆分后保持等价。
4. **C4 的对照口径已钉死**：R8 重derive 组（prompt P50 17,076→目标 ≤13,661、
   延迟 19.8s、judge 均分 0.9602−0.038、PASS ≥42/44，见交接文档任务 C 回写块）。
   本笔记的 7,094/42% 是诊断数据，不是门槛口径。

## 四、复现

```bash
# 零模型调用；Qdrant 文件锁互斥——探针强制独立 VECTOR_STORE_DIR（脚本内置）
uv run python scripts/eval/skill_token_breakdown.py --report eval/report-20260905-142017.json
uv run pytest tests/test_skill_token_breakdown.py -q   # 纯函数守卫 11 条
```
