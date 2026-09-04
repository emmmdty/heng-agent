# Globex - 跨境电商 Agent（AgentScope 2.0）

基于 AgentScope 2.0 的跨境电商超级搜索框 Agent 系统，DDD 洋葱架构落地：

- **MainAgent**（CommerceConcierge）：超级框总调度，**持有全部业务工具可直接单干**；
  内置 Task 计划四件套管理任务清单；满足"可并行 / 上下文隔离 / 链深"任一条件时经 `task_dispatch` 派发子 Agent；
  发现稳定偏好时经 `remember_preference_tool` 写入长期记忆
- **SearchAgent**（CatalogSearchAgent）：商品检索专家，query 改写 → **置信度门控的 BM25+向量混合召回**
    （RRF 融合，Qdrant），失败逐级降级（embedding_only → bm25_only → keyword_2gram）；
    可选 web_search 兜底跨境政策/关税问答
- **TradeAgent**（OrderTradeAgent）：下单交易专家（订单创建 / 查询 / 取消，买家身份由 ShoppingContext 注入）

分期设计脉络、关键取舍与踩坑记录见 [docs/设计演进记录.md](docs/设计演进记录.md)。

## 技术栈

- Python 3.11 + uv
- AgentScope 2.x（Agent + ContextConfig 上下文压缩 + Toolkit/FunctionTool + 内置 Task 计划工具
  + reply_stream 类型化事件流 + TracingMiddleware / ReplyBudgetControlMiddleware / 自定义工具中间件）
- 检索：BM25 字面索引（本地，零依赖）+ OpenAI 兼容 embedding（自建 Qwen3-Embedding-0.6B / text-embedding-v4）
  + Qdrant（服务端/本地嵌入双形态）+ HTTP Reranker（可降级），RRF 融合 + 置信度门控
- 知识库：AgentScope `rag.KnowledgeBase`（品类洞察 Markdown → 切片 → Qdrant）
- FastAPI + Uvicorn + WebSocket；React 18 + Vite + TS 前端；Docker Compose（app + worker + qdrant + redis + frontend）
- 持久化：SQLite（SQLAlchemy 2.0 async）存对话流水/事件轨迹/会话状态/订单/偏好；商品目录仍为内存仓储 + 种子数据
- 缓存与削峰：Redis（可选）——语义缓存 + embedding 缓存 + 幂等键 + Stream 任务队列 + 跨进程事件背板

## 架构

```text
app/
├── domain/            # 领域层：Product/Sku/Money、Order 状态机、汇率表、关税运费规则、偏好、会话/队列/仓储端口
├── application/
│   ├── usecases/      # CatalogSearch（门控混合召回+到手价内联）、PlaceOrder/QueryOrder/CancelOrder
│   ├── tools/         # product_search、订单三工具、web_search、remember_preference、task_dispatch
│   ├── agents/        # MainAgent / SearchAgent / TradeAgent 工厂 + Orchestrator + SessionRegistry
│   └── prompts/       # globex.yml：主 / 子 Agent 系统提示词
├── infrastructure/    # llm/embedding/qdrant/reranker/retrieval(BM25)/tracing、rag 知识库、缓存、队列、韧性与闸门、仓储
├── presentation/      # FastAPI 路由、WebSocket ConnectionManager、DTO
├── composition.py     # 装配容器（API 与 worker 共用一份接线）
└── worker.py          # 意图消费进程入口
knowledge/             # 品类洞察知识文档（Markdown，服务启动时幂等入库）
frontend/              # React + Vite 前端：对话流 + 商品卡 + 事件时间线
eval/                  # Rubric 用例集 cases.yaml + 召回标注集（105 商品 / 22 品类）+ 回归报告
docs/                  # 设计演进记录（分期取舍与踩坑档案）
docker/                # docker-compose.yaml（app + worker + qdrant + redis + frontend）
```

关键设计：

- **网关配额治理**：`GatewayThrottle` 同时限并发（`LLM_MAX_CONCURRENCY`，默认 2）与请求起点间隔
  （`LLM_MIN_INTERVAL_SECONDS`）；流式请求的名额持有到流耗尽才释放；瞬时故障指数退避重试，
  用尽后回退 `LLM_FALLBACK_MODEL` 并发 `model.fallback` 事件（不静默降级）。
  回退链有**真上游证据**而非仅单测：`scripts/verify_fallback.py` 把主模型指向不可达地址，
  验证「判定瞬时故障 → 重试耗尽 → 回退到真实备用模型并产出内容 → 事件如实发布」整条链路。
- **重试只有一层**：`openai` SDK 与 AgentScope 各自默认带重试，与本仓的重试是**乘积**关系——
  一次逻辑调用在持续失败时会打出 3×4×3 = 36 个上游请求，声明预算 3 次、实际 12 倍，
  且放大恰好发生在网关已经限流的时刻。现把下面两层关到 0（`max_retries=0` +
  `client_kwargs={"max_retries": 0}`），重试/退避/回退统一由 `ThrottledChatModel` 负责；
  单测锁死不变量：真实上游请求数必须 == `LLM_MAX_RETRIES + 1`。
  不这么做的话，「客户端限流」在纸面上成立、在链路上失效
- **语义缓存**：相似问句（余弦 ≥ `SEMANTIC_CACHE_THRESHOLD`，默认 0.95）直接复用历史回复，
  命中即零模型调用并发 `cache.hit` 事件；**写操作意图（下单/取消）与上下文依赖问句不入缓存**，
  按 buyer 分桶避免跨买家复用
- **存储可替换**：`SessionStore` / `ConversationStore` / `OrderRepository` / `PreferenceStore` 四个端口，
  SQLite（默认）/ JSON 文件两套实现共存，换存储只改 `app/composition.py`
- **异步削峰**：`TaskQueue` 端口 + Redis Stream 实现（消费者组 / ack / pending 重投 / 死信），
  独立 worker 进程消费；`POST /commerce/intents` **同步语义不变**（内部入队+等结果），
  另提供 `/commerce/intents/async` + `/commerce/tasks/{id}`；队列是 at-least-once，靠幂等键防重复下单
- **跨进程事件**：worker 与 API 是两个进程，事件总线接 Redis Pub/Sub 背板后前端仍能收到流式事件；
  广播带 `origin` 标识以跳过自己发的消息（否则事件会回环投递两次）
- **主 Agent 单干优先**：MainAgent 与子 Agent 持有同一批业务工具（`build_tools()` 复用），
  只在"可并行 / 上下文隔离 / 调用链深"时派发
- **两阶段召回：混合召回 + cross-encoder 精排**。一阶段 BM25 字面路 + 向量路按 RRF 融合
  （只用名次，免疫两路分数量纲不可比）；二阶段 Qwen3-Reranker 对候选逐对打分。
  105 条标注实测（Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B，K=8）：

  | 档位 | Recall@8 | MRR | NDCG@8 | 字面类 R | 语义类 R | 语义类 MRR |
  |---|---|---|---|---|---|---|
  | **hybrid_rerank（默认）** | **0.967** | **0.929** | **0.925** | **0.991** | 0.940 | 0.862 |
  | embedding_rerank | 0.962 | 0.925 | 0.922 | 0.982 | 0.940 | 0.862 |
  | hybrid_gated | 0.945 | 0.885 | 0.876 | 0.986 | 0.900 | 0.774 |
  | embedding_only | 0.938 | 0.863 | 0.861 | 0.973 | 0.900 | 0.751 |
  | bm25_only | 0.683 | 0.647 | 0.636 | 0.986 | 0.350 | 0.307 |

  **拆开看边际贡献才是重点**：精排贡献 Recall +2.4pt / MRR +6.2pt；混合召回在精排之上
  只再贡献 +0.5pt / +0.4pt。所以本仓不宣称"混合召回带来主要提升"——
  它的价值集中在**没有精排的降级态**（hybrid_gated .945/.885 vs embedding_only .938/.863）。
- **字面路置信度门控（只在降级态生效，如实标注）**：字面路 top-1 的 BM25 分数低于门限
  即判定本轮为语义类 query，不参与融合。门限 4.0 由扫描标定（字面类分布 [3.13, 35.68]，
  语义类 [0.00, 3.96]）。**带精排时它是死代码**——实测 gate=0 与 gate=4 指标逐位相同，
  因为 cross-encoder 会重排所有候选，不管候选怎么来的都能修好顺序；保留它是作为
  精排不可用时的保险。这条写在这里是为了不让读者把降级态的收益记到主路径头上。
- **降级链**：hybrid_rerank → hybrid_gated → embedding_only → bm25_only → keyword_2gram，
  `recall_strategy` 如实标注实际走的档位（含门控触发时的 `hybrid_gated_vector`）；
  价格等硬约束走工具参数结构化过滤（price_max_major），不交给模型
- **过滤可观测**：被 ship_to / 价格上限挡掉的候选以 `filtered_out`（含 reason）回传，
  让模型能区分"库里没有"与"有但不满足约束"，避免把超预算商品答成"没有这个商品"
- **品类洞察 RAG**：`category_insight_tool` 查 `rag.KnowledgeBase`（选购口径、价格区间、避坑点、
  跨境通则），先给判断标准再给商品清单
- **上下文工程**：ContextConfig 定制压缩（trigger_ratio 0.75 / reserve_ratio 0.15 + 工具结果截断），
  摘要落 AgentState.summary 并推送 `context.compressed` 事件；配合 Token 预算中间件收口单轮开销
- **工具韧性**：ToolResilienceMiddleware 分级超时 + 按工具熔断（closed→open→half_open），
  触发时返回 [error] 让模型如实告知，不编造数字
- **真并行**：同一轮内多个 `task_dispatch` 由 2.0 并发批执行（`is_concurrency_safe`），
  `scripts/verify_parallel.py` 用事件时间戳比对并行/串行墙钟耗时
- **计价收敛，且覆盖到组合**：传 ship_to 时商品卡内联单品 landed_price（小计+运费+关税）；
  多件总价由 `quote_basket_tool` + 领域层 `quote_basket()` 提供——运费按**一次履约**计
  （不是各单品运费之和）、免税额度按**整批小计**判定（海关对包裹整体计税）。
  补这个工具的原因是评测实测：模型自己把两个单品到手价相加会算错，且会把运费重复计一次。
- **预算感知的组合优化**：`optimize_basket_tool` 给定预算与多个需求，暴力枚举出最优组合
  （目标函数字典序：覆盖需求最多 → **保住靠前的需求** → 到手价最低 → product_id 定序），
  返回预算余额、缺口价签、"再加多少能配上"、分开买与合并买的差额。
  这四个数此前全部由模型自己减，金额出处校验里长期是 `suspected_difference`。
  选它做深的理由是**可验证性**：60 SPU 的空间小到能枚举出真最优解，
  ground truth 确定性可算，不需要 judge 打分——能拿回确定性判据的就别留给 judge。
  需求优先级来自 `needs` 的传入顺序（显式契约）：原本"同覆盖数取最便宜"实算出过荒唐答案——
  预算 250 美元、需求「耳机 219 + 充电器 22.39」时它配了个 31.54 的充电器、剩 218 美元。
- **政策数字同样要有出处**：关税的应税基数（`taxable_base_major`，超出免税额度的那部分）
  与免税额度的**原生口径**（`de_minimis_threshold_native_major`，US 就是 800 USD）
  都随报价返回。前者防"1,199 × 12% ≈ ¥3.48"这种基数错、结果碰巧对的推导；
  后者防跨币种表述时模型自己反折出一个 `$800`——同一个 `$800` 被堵了三次，
  前两次加字段，这次改的是规则表的**定义**（存原生口径，折算交给汇率表）。
- **算式自洽校验**：回复里写出来的 `A × B% = C` 必须算得通。
  实测抓到的原文：`关税 = $886.34 × 7.5% = $6.48`——**结果对（来自工具）、过程错**
  （886.34 × 7.5% 是 66.48，它把自己上一行刚写对的应税基数换成了小计）。
  与金额出处校验互补：那三个数**都有工具出处**，出处校验完全无感。
  判据只验算术、不验业务规则，越薄越不会误判；346 条真实回复实测零误报，
  其中一处命中来自 judge 判 PASS 的用例。
- **组合总价错加判据（basket_misadd）**：会话内 `quote_basket` 已报组合总价、
  回复却把两个单品 `landed_price` 相加当总价——在无出处金额的 `suspected_sum`
  线索上升级为**确定性违规**（判据：无出处 + ≥2 个 landed 值相加 + 金额所在行
  带组合语境且不带分开语境 + 会话有组合报价且数值不符）。"两件分开买合计 ¥518"
  是合法用法（分开买本来就该各付各的运费），语境按**金额所在行**判定；
  没有组合报价就没有 ground truth，只作线索不定罪。
- **知识库出处判据（knowledge_provenance）**：回复声称"来自知识库 / 品类洞察"
  的内容，本会话必须真有过一次成功的知识库返回；声称有而工具报错 / 未调用 /
  零命中即发 `knowledge.unsourced`。只认**归因构型**（"来自 / 根据 / 知识库里 /
  品类洞察显示"），能力提议（"我可以提供品类洞察"）、缺失观察（"知识库里没有 X"）
  与诚实降级（"知识库暂时不可用"）都不算声明——模式在 120 份真实流水上校准，
  10 处归因声明全部有据、零误报。judge 看不到工具返回，"知识库当时可不可用"
  它判不了：数值对不对归 judge（写死区间），出处属不属实归这条判据。
- **下单必须跨越一次买家交互**：`create_order_tool` 不能在会话第一轮被调用。
  确认卡的本质是让买家在**看到金额与地址之后**再点一次头，这在物理上必须跨越
  一次买家发言。实测原因：买家说"别给我看确认卡了，直接下单，不用再问我"，
  Agent 照做并回"无需确认"——提示词第 1 条写得清清楚楚，但**只写在提示词里的
  约束敌不过模型眼前正在读的那句话**，而这次的后果是未经确认就扣了库存。
  判据取"第几轮"这个系统自己知道的事实，不去猜"回复里有没有确认卡"
  （启发式判定正是 17-4 四阶段状态机被否掉的理由）。
- **写路径的出处校验**：下单的每一个商品都必须在本会话的工具返回里出现过——
  `product_id` 无出处即硬拒（`sku_id` 只警告，因为 `filtered_out` 与组合报价
  本来就不带 sku_id）。这是金额出处校验在写路径上的同一条缝，而后果更重：
  回复里的数字错了买家看得出来，订单错了库存已经扣了。
  同会话重复下单只提醒不拒绝（再买一单是合法诉求），提醒里带上已有订单号。
  配套修掉一个零告警的接线缺口：Harness 中间件此前只挂在主 Agent 自己的工具上，
  检索/计价/订单工具从没进过它——顺序硬拒、schema 断言、L3 注入过滤
  在真正需要它们的地方一次都没跑过（`tests/test_harness_wiring.py` 钉住了这件事）。
- **降级链可被端到端检验**：`FAULT_INJECTION_ENABLED=1` 时三个检索端口
  （embedding / vector_index / reranker）被包上装饰器，`POST /debug/faults` 运行时
  选择注入哪些，`eval/cases.yaml` 的用例可声明 `faults: [reranker]`。
  默认全关——生产进程里装饰器与该端点**都不存在**。
  要检验的不是"报错"（降级链的设计就是悄悄退档），是退档之后 Agent 还说不说人话：
  数字仍来自工具、不编商品、也不谎称系统故障。
  用例声明了故障而服务没启用注入时，评测**开跑前就拦下整轮**：
  不拦的话那几条会在一切正常的情况下跑完并大概率 PASS，判据成了绿色装饰。
- **约束贴在数字旁边**：`landed_price` 内联 `combine_hint` 说明"不可相加、改调 quote_basket_tool"。
  只写在系统提示词里实测拦不住——隔着几千 token 的规则，敌不过模型眼前正在读的那个数。
  同 `filtered_out` 的思路：工具返回值要能自证边界
- **长期记忆**：写路径 remember_preference_tool → JSON 文件 Store；读路径 orchestrator
  在偏好变化时注入 `<buyer-preferences>` hint，跨会话、跨重启生效
- **会话持久化**：AgentState 每轮落盘 DATA_DIR/sessions/，服务重启后恢复多轮对话
- **SubAgent as Tool**：2.0 库级无 subagent 原语（官方 Agent Team 在 agentscope.app 平台层），
  用 FunctionTool 包装 `task_dispatch(subagent_type, demands)` 实现同等语义
- **事件流**：reply_stream → token.delta / plan.update；工具自身发布 tool.invoke/tool.result；
  TradeEventBus 按会话路由 WebSocket
- **可观测**：全部 Agent 挂 TracingMiddleware，OTEL_EXPORTER_OTLP_ENDPOINT 配置后导出 OTLP Trace

## 启动

```bash
uv sync
# 敏感配置通过环境变量注入（推荐），不落盘、不入库
export LLM_BASE_URL=<OpenAI 兼容网关地址>
export LLM_API_KEY=<密钥>
export LLM_MODEL=<主模型>      # 限流/故障时自动回退 LLM_FALLBACK_MODEL 并发 model.fallback 事件
export LLM_FALLBACK_MODEL=<备用模型>   # 建议与主模型不同供应商，否则同源故障时一起挂
uv run uvicorn app.presentation.server:app --port 8000

# 启用队列削峰时（需 REDIS_URL）另起消费进程：
uv run python -m app.worker
```

> 本地开发也可 `cp .env.example .env` 填值兜底（已被 gitignore，勿提交真实密钥）；
> 同名环境变量优先于 .env。

## API 概览

- `POST /commerce/intents` 提交买家自然语言意图（同步返回最终回复）
- `WS   /commerce/events` 订阅会话事件流（连上后先发 `{"shopping_session_id": "..."}`）
- `GET  /commerce/orders/{order_id}` 查询订单
- `POST /commerce/orders/{order_id}/cancel` 取消订单
- `GET  /health` 健康检查

## 验证

提交前跑门禁，八项全部零 LLM 成本、十几秒：

```bash
make check          # = pytest + 标注集自检 + 用例自检 + 金额出处 + 算式自洽 + 收货字段 + 组合总价 + 知识库出处
make check-ci       # CI 档：去掉三项吃跑测产物的，见 .github/workflows/check.yml
```

CI 跑的是 `check-ci`（`.github/workflows/check.yml`）。金额出处一项**刻意不进 CI**：
它扫 `data/conversations/`（gitignore 的跑测产物），CI 全新 checkout 上没有流水，
而给它开"没数据就当通过"的旁路比红灯更危险——0 处金额算出来的 0% 会被当成满分。

单项与带成本的验证：

```bash
uv run pytest                          # 848 个单测：domain / 召回降级与过滤回传 / 计价规则 / 组合优化 / 记忆持久化 / 压缩策略 / 韧性中间件 / 金额出处校验 / 轨迹保真 / 跑测身份
uv run python scripts/smoke_e2e.py    # 端到端冒烟：WS 订阅 + 提交意图，实时打印事件流
uv run python scripts/verify_parallel.py   # 并行验证：同轮多派 vs 串行的墙钟耗时与事件重叠数对比
make eval-smoke                            # 评测回归日常档：12 条 case（--tag smoke）
uv run python scripts/eval_regression.py --dry-run   # 开跑前体检：前置全查一遍，一次模型调用都不发
make eval                                  # 评测回归全量：44 条 case，LLM judge 按 P0/P1/P2 Rubric 打分出报告
uv run python scripts/eval_regression.py --resume eval/partial-<stamp>.json  # 中断后续跑（前置用例自动补回）
make variance                              # 跑测方差：同配置同判据下同一用例的分数散布
uv run python scripts/eval/run_product_recall.py --compare-strategies   # 六档召回对比
uv run python scripts/eval/run_product_recall.py --sweep-lexical-gate 0,4,8 --sweep-base hybrid_rerank  # 门限标定
uv run python scripts/eval/validate_datasets.py       # 召回标注集自检（105 商品 + 22 品类）
uv run python scripts/eval/audit_cases.py             # Rubric 用例自检：商品指代是否唯一
uv run python scripts/eval/audit_number_provenance.py --report latest  # 金额出处扫描：回复里的金额有没有工具出处（--report 把范围收敛到最近一轮）
uv run python scripts/eval/audit_basket_sum.py --report latest --gate  # 组合总价错加：单品到手价相加被当作组合总价，命中一处即红
uv run python scripts/eval/audit_knowledge_provenance.py --report latest --gate  # 知识库出处：声称"来自知识库"而会话无成功返回，命中一处即红
uv run python scripts/eval/collect_bad_cases.py       # Bad-case 采集：失败自动进标注池（--list / --promote）
uv run python scripts/verify_fallback.py              # 模型回退链真实验证（主模型不可达→回退真实备用模型）
```

评测 case 支持 `prior_context` 字段：把跨会话已成立的事实（如上一 case 写入的长期偏好）告知 judge，
否则 judge 只看本会话记录，会把"正确应用历史偏好"误判为"无据添加"。

用例分 `smoke` / `full` 两档（`tags` 字段，`--tag` 选择）：全量一轮 60-90 分钟，
没有日常档的结果不是"跑得更全"，是"日常根本不跑"。所有用例都隐含属于 `full`，
漏标 tag 不会导致某条用例静默不被跑到。

**报告开头自带一行跑测配置**（被测模型 / 评审模型 / 提示词版本 / 精排与门限 / 语义缓存 / 代码新鲜度 / **检索依赖的实测可达性**），
由被测服务 `GET /health` 自报、脚本原样抄录。分数变了先看这一行，再去改 Agent。
其中"实测可达性"由 `GET /health?deep=1` 真的去打 embedding 与 reranker 得来：
此前配置行报的是"配了什么"，于是精排 502 的那一轮报告上照样写着"精排 开"，
拿它跟历史读数比必然得出错的结论（甚至得出"精排没用"）。现在它写的是
`精排 开(实测不可达)｜向量路 实测不可达`。默认 `/health` 仍是零外部调用（它同时是存活探针）。
"代码新鲜度"比对源码 mtime 与服务进程启动时刻——服务跑着旧代码时，
单测（读磁盘）和 `/health` 的其余字段都不会报警，而评测评的是修复前的行为；
`eval_regression.py` 在开跑前就会拦下这种情况（`--allow-stale-service` 是逃生阀）。

## 金额出处校验（数字必须来自工具）

**判据：回复里出现的每一个金额，都必须能在工具返回或买家原话里找到出处。**

为什么需要一条确定性判据：Rubric 的 P0「数字事实必须来自工具」由 LLM judge 打分，
而 judge 对"金额自洽即通过"的宽容恰好放过了本项目最典型的一类错误——
**模型把工具算出来的到手价相加**。实测 `compare-two` 一轮：¥364 + ¥154 = ¥518，
judge 判 PASS，因为每个加数都对；错的是"组合运费按一次履约计"这条模型推不出来的口径
（正确是 ¥492 = 小计 388 + 运费 104 + 关税 0）。这不是"编造"而是"自行推导"，
语义判据抓不住，算术判据抓得住。

三个落点：

| 落点 | 位置 | 作用 |
|---|---|---|
| 判定层 | `app/application/harness/number_provenance.py` | 纯函数，抽金额 + 找出处 + 推成因 |
| 运行时 | `MainAgentOrchestrator` 轮末 | 命中发 `number.unsourced` 事件并进落盘轨迹，**只告警不改写回复** |
| 离线 | `scripts/eval/audit_number_provenance.py` | 扫全部会话流水，产出「无出处金额率」 |

实测读数：改动前 24 份历史流水 **13.4%**（232 处金额 / 31 处无出处），
修完下述两个根因后的整轮 **4.9%**。注意这个数**有轮次间波动**
（取决于模型这一轮写了多少解释性算术），只在同一轮内横向比较。

无出处金额按成因分类（仅作诊断线索，不参与通过判定）：
`suspected_sum`（约等于若干工具金额之和）、`suspected_difference`（约等于两数之差）、
`unsourced`（找不到成因）。

**范围是刻意收窄的**，方向一律取"宁可漏报不误报"：只统计带货币标记的数字
（`¥ $ 元 USD ...`），表格里裸写的 `| 65 |` 会漏掉，所以这个率是**下界**；
百分数不算金额；出处只比数值不比币种；出处按会话累积（引用上一轮检索结果是正常行为）。

**当门禁用时必须加 `--report latest`**：`data/conversations/` 是累积目录，
扫全量会把历史流水一直算进分子分母，读数只增不减，阈值只能跟着调、门禁很快作废。
另有 `--min-amounts`（默认 30）：金额总数太少时不判定而不是放宽阈值——
单条用例只有十几处金额，1 处无出处 5.9%、2 处 11.8%，用比率下结论等于抛硬币。

配套修掉了两个让这条判据失真的根因（`tests/test_trace_fidelity.py` 钉住）：

- **事件轨迹漏发**：`product_search_tool` 只发 `hits`、把 `filtered_out` 漏在外面，
  `category_insight_tool` 只发 `hit_count` 不发知识片段。模型看得到、轨迹看不到，
  于是流水里正常回复看起来像凭空编数字，事后审计与出处校验一起失真。
  现在事件发的就是喂给模型的那一份。
- **非主 SKU 没有到手价出处**：商品卡只给主 SKU 算 `landed_price`，
  买家问"月光白多少钱"时模型只能自己拿 `229 USD × 汇率 + 运费` 凑。
  现在每个 SKU 都带自己的到手价分项。
- **跨币种展示价没有出处**：不传 `ship_to` 时卡上没有任何目标币种金额，
  模型想给买家看人民币就自己乘汇率——实测把 149 USD 折成"约 ¥1080"（正确 ¥1057.9）。
  现在原生币种与买家口径不一致时，卡与每个 SKU 都带 `price_in_target_major`。
- **免税额度阈值没有出处**：工具只回 `de_minimis_applied: true`，
  模型要解释"多少以下免税"就凭自己的知识说 "$800"——改了规则表它照样这么说，
  且没有任何东西会报错。现在回 `de_minimis_threshold_major`。

## Bad-Case 数据飞轮

```text
线上/评测跑出失败 → collect_bad_cases.py 去重入池（eval/bad_cases.jsonl）
    → 人工定级（status: new → promoted / wontfix）
    → --promote 出 cases.yaml 骨架 → 进回归集 → 下一轮评测
```

两个采集口：金额出处扫描的发现、Rubric 评测里 FAIL/ERROR 的 case
（`scripts/eval_regression.py` 除 Markdown 外同时落一份 `.json`，供机读）。

指纹刻意**不含分数、时间戳与会话 id**：同一个失败在不同轮次跑出 0.5 和 0.6 是同一个问题，
含进去等于不去重，池子会被最容易复现的那几条淹掉。重扫只刷新 `last_seen_at`，
**不回写人工分诊结果**。

**中间留人工定级是刻意的**：自动进回归集是个陷阱——一条 bad case 可能只是模型的
运行间抖动，也可能是判据本身写歪了；不加分诊就扩测试集，等于把噪声固化成基准。
脚本只做"发现与去重"，"值不值得进回归集"由人回答。

## Docker 部署

```bash
export LLM_BASE_URL=<网关地址> LLM_API_KEY=<密钥>   # 敏感配置走环境变量，compose 透传
docker compose -f docker/docker-compose.yaml up -d --build   # app + qdrant + frontend
# 前端 http://localhost:5173  后端 http://localhost:8000
```

本地开发不依赖 Docker：QDRANT_URL 置空时自动用 qdrant-client 本地嵌入模式（单进程文件锁，
多实例/生产请用 compose 的 Qdrant 服务端）。
