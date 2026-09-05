# 「衡 · Heng」提交前门禁。
#
# check 里的六项全部是**零 LLM 成本**的确定性检查，加起来十几秒，
# 所以约定是「每次提交前必跑」，而不是「想起来再跑」。
# 真正烧钱的端到端 Rubric 回归（eval / eval-smoke）不在 check 里，单独手动跑。
#
# make 遇到非零退出码即中断，六个脚本的退出码都已对齐（0 通过 / 1 不通过）。

.PHONY: check check-ci test datasets cases provenance arithmetic contact basket knowledge eval eval-mainline eval-smoke variance cost health serve serve-faults

check: test datasets cases provenance arithmetic contact basket knowledge
	@echo "== 门禁通过 =="

# CI 档：check 去掉三项吃跑测产物的（金额出处、算式自洽、收货字段）。
#
# provenance 扫的是 data/conversations/（跑测产物，gitignore），
# CI 全新 checkout 上一份流水都没有，`--report latest` 会按设计报错退出。
# **不给它加"没数据就当通过"的旁路**：0 处金额算出来的 0% 被当成满分放行，
# 比红灯更危险（十期踩坑 33 的同一条）。宁可在 CI 里明确少跑一项，
# 也不要让一项判据在 CI 上变成永远绿的装饰。
#
# 另：这三项虽然零 LLM 成本，却仍需要 LLM_API_KEY **存在**——
# 有 4 条测试走 load_settings()（它对缺密钥是 fail-fast 的），一个都不打上游。
# 所以 CI 里注入占位值即可，见 .github/workflows/check.yml。
check-ci: test datasets cases
	@echo "== CI 门禁通过（金额出处 / 算式自洽 / 收货字段需本地跑测产物，见 check）=="

test:
	uv run pytest -q

datasets:
	uv run python scripts/eval/validate_datasets.py

cases:
	uv run python scripts/eval/audit_cases.py

# 阈值 = 当前基线 + 余量，不能定成 0：
# 对已有出处数字的对比与修辞取整属于合理用法，本来就会占掉几个点。
# 当前基线 6.7%（九期实测），收掉免税额度阈值那条尾巴后预计更低，
# 届时再往下收紧，不要在拿到新读数之前先收。
PROVENANCE_MAX_RATIO ?= 0.08
# 吃跑测产物的门禁默认看最近一轮（--report latest）。补跑/定向切片会把 latest
# 顶成小样本，那时用 REPORT=<报告路径> 钉死要判的那一轮——引用基线一律钉文件名，
# 这是同一纪律的门禁侧入口（不改变默认行为，不加任何"没数据当通过"的旁路）。
REPORT ?= latest

# --report latest 把范围收敛到最近一轮：data/conversations/ 是累积目录，
# 扫全量会把历史流水的无出处金额一直算进来，每跑一轮读数就往上抬一点，
# 阈值只能跟着调，门禁很快就废了。
provenance:
	uv run python scripts/eval/audit_number_provenance.py --report $(REPORT) --max-ratio $(PROVENANCE_MAX_RATIO)

# 算式自洽：回复里显式写出的 `A × B% = C` 得算得通。
#
# 与 provenance 并列但**口径不同：不设阈值、不设样本量下限，命中一处即红**。
# 无出处金额率是比率（修辞取整本来就占几个点），小样本不判定是对的；
# 而 886.34 × 7.5% = 6.48 是能指着原文说"这一行算错了"的事实错误，
# 没有"这轮抖了一下"的解释空间，也就没有摊薄它的口径（踩坑 45 同一面）。
arithmetic:
	uv run python scripts/eval/audit_arithmetic.py --report $(REPORT) --gate

# 收货字段出处：回复里的地址 / 电话 / 邮编，买家没给过、工具没返回过就不许出现。
#
# 口径同算式自洽（不设阈值、不设样本量下限，命中一处即红），理由也同一条：
# "编造了一个收货地址"是能指着原文说"这个地址不存在"的事实错误，
# 不是可以被样本量摊薄的比率。二十期 `clarify-missing-address` 那次，
# Agent 写的是"您之前的记录是上海市浦东新区世纪大道100号"——**一个金额都没有**，
# 前两条扫描完全无感，所以必须是第三条独立判据。
contact:
	uv run python scripts/eval/audit_contact_provenance.py --report $(REPORT) --gate

# 组合总价错加（basket_misadd）：单品到手价相加被当作组合总价，即运费重复计。
#
# 口径同算式自洽、收货字段：不设阈值、不设样本量下限，命中一处即红——
# "这行把组合总价算错了"是能指着原文说的事实错误。
# 但注意它有一条前置：会话内存在 quote_basket 报价才有 ground truth，
# 没有报价时判据只作线索、不判罪；所以"0 违规"分两种，
# 判词会写明是"判过了、全对"还是"压根没东西可判"（踩坑 33）。
basket:
	uv run python scripts/eval/audit_basket_sum.py --report $(REPORT) --gate

# 知识库出处：声称"来自知识库 / 品类洞察"的内容，本会话必须真有过成功返回。
#
# 口径同算式自洽、收货字段、basket：不设阈值、不设样本量下限，命中一处即红。
# "知识库根本没返回过，回复却说'知识库里说'"是能指着原文说的张冠李戴。
# 判据刻意窄：只认"知识库 / 品类洞察"字样的归因构型，能力提议、缺失观察
# （"知识库里没有 X"）与诚实降级（"知识库暂时不可用"）都不算声明。
knowledge:
	uv run python scripts/eval/audit_knowledge_provenance.py --report $(REPORT) --gate

# —— 以下带真实 LLM 成本，不进 check ——

# 整轮回归：40 条 ≈ 80-120 分钟。跑之前先确认 /health 里 semantic_cache 为 false。
# 其中 3 条带故障注入，需要服务以 make serve-faults 起（否则开跑前被拦下）。
eval:
	EVAL_JUDGE_MODEL=longcat-2.0 uv run python -u scripts/eval_regression.py

# 主线档：全部 55 条剔掉 11 条红队 = 44 条，与二十三期 R7 基线**同一份考卷**
# （用例集身份 d9e463d2，已与 eval/report-20260904-180527.json 逐位核对）。
#
# 为什么要单独一档：red team 用例是二十三期加的，而 `full` 是隐含标签
# （所有用例都属于它），于是 `make eval` 从那以后跑的是 55 条。
# 交接文档与贡献证明里"44 条 44/44、均分 0.993"这条基线一度**没有选择器能复现**，
# 而二十五期 A/B 的一票否决护栏引用的正是它——拿 55 条那轮去比就是两把尺子。
eval-mainline:
	EVAL_JUDGE_MODEL=longcat-2.0 uv run python -u scripts/eval_regression.py --exclude-tag redteam

# 冒烟档：日常改代码只跑这一档。
eval-smoke:
	EVAL_JUDGE_MODEL=longcat-2.0 uv run python -u scripts/eval_regression.py --tag smoke

# 跑测方差：同一配置 + 同一判据下，同一条用例的分数散布。
# 没有方差就没有显著性——"这次 0.973、上次 0.95"是改好了还是抖了一下，
# 不量方差只能靠感觉答，而靠感觉答的结果是真退化被当成抖动放过。
variance:
	uv run python scripts/eval/variance.py

# token 成本 / 轮延迟：读流水的 usage 与 latency_ms（二十三期清单 2）。
# 零模型成本（只读已落盘的流水），不进 check——它是报告不是判据。
# usage 字段自二十三期起才有；旧流水只出延迟读数，token 覆盖面会被点名。
cost:
	uv run python scripts/eval/audit_cost_latency.py --report latest

# 外部依赖体检：两条隧道 + 应用。读数不对时先查这里，再去看分数。
health:
	@curl -s -m5 -XPOST http://127.0.0.1:11436/v1/embeddings \
	  -H 'Content-Type: application/json' -d '{"model":"q","input":"x"}' | head -c 60; echo "  <- embedding"
	@curl -s -m5 http://127.0.0.1:11437/health; echo "  <- reranker"
	@curl -s -m10 "http://127.0.0.1:8000/health?deep=1"; echo "  <- app（deep=1 会真的去探两条隧道）"

serve:
	uv run uvicorn app.presentation.server:app --port 8000

# 带故障注入的服务：eval/cases.yaml 里 `faults:` 那几条用例需要它。
# 没有它，声明了故障的用例会在开跑前被 eval_regression 拦下
# （不拦的话它们会在一切正常的情况下跑完并大概率 PASS——判据成了绿色装饰）。
# **生产不要这么起**：装饰器与 /debug/faults 端点都会存在。
serve-faults:
	FAULT_INJECTION_ENABLED=1 uv run uvicorn app.presentation.server:app --port 8000
