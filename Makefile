# Globex 提交前门禁。
#
# check 里的四项全部是**零 LLM 成本**的确定性检查，加起来十几秒，
# 所以约定是「每次提交前必跑」，而不是「想起来再跑」。
# 真正烧钱的端到端 Rubric 回归（eval / eval-smoke）不在 check 里，单独手动跑。
#
# make 遇到非零退出码即中断，四个脚本的退出码都已对齐（0 通过 / 1 不通过）。

.PHONY: check check-ci test datasets cases provenance eval eval-smoke variance health serve serve-faults

check: test datasets cases provenance
	@echo "== 门禁通过 =="

# CI 档：check 去掉金额出处那一项。
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
	@echo "== CI 门禁通过（金额出处项需本地跑测产物，见 check）=="

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

# --report latest 把范围收敛到最近一轮：data/conversations/ 是累积目录，
# 扫全量会把历史流水的无出处金额一直算进来，每跑一轮读数就往上抬一点，
# 阈值只能跟着调，门禁很快就废了。
provenance:
	uv run python scripts/eval/audit_number_provenance.py --report latest --max-ratio $(PROVENANCE_MAX_RATIO)

# —— 以下带真实 LLM 成本，不进 check ——

# 整轮回归：40 条 ≈ 80-120 分钟。跑之前先确认 /health 里 semantic_cache 为 false。
# 其中 3 条带故障注入，需要服务以 make serve-faults 起（否则开跑前被拦下）。
eval:
	EVAL_JUDGE_MODEL=longcat-2.0 uv run python -u scripts/eval_regression.py

# 冒烟档：日常改代码只跑这一档。
eval-smoke:
	EVAL_JUDGE_MODEL=longcat-2.0 uv run python -u scripts/eval_regression.py --tag smoke

# 跑测方差：同一配置 + 同一判据下，同一条用例的分数散布。
# 没有方差就没有显著性——"这次 0.973、上次 0.95"是改好了还是抖了一下，
# 不量方差只能靠感觉答，而靠感觉答的结果是真退化被当成抖动放过。
variance:
	uv run python scripts/eval/variance.py

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
