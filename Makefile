# Globex 提交前门禁。
#
# check 里的四项全部是**零 LLM 成本**的确定性检查，加起来十几秒，
# 所以约定是「每次提交前必跑」，而不是「想起来再跑」。
# 真正烧钱的端到端 Rubric 回归（eval / eval-smoke）不在 check 里，单独手动跑。
#
# make 遇到非零退出码即中断，四个脚本的退出码都已对齐（0 通过 / 1 不通过）。

.PHONY: check test datasets cases provenance eval eval-smoke health serve

check: test datasets cases provenance
	@echo "== 门禁通过 =="

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

# 整轮回归：13 条 ≈ 25-40 分钟。跑之前先确认 /health 里 semantic_cache 为 false。
eval:
	EVAL_JUDGE_MODEL=deepseek-v4-flash uv run python -u scripts/eval_regression.py

# 冒烟档：日常改代码只跑这一档。
eval-smoke:
	EVAL_JUDGE_MODEL=deepseek-v4-flash uv run python -u scripts/eval_regression.py --tag smoke

# 外部依赖体检：两条隧道 + 应用。读数不对时先查这里，再去看分数。
health:
	@curl -s -m5 -XPOST http://127.0.0.1:11436/v1/embeddings \
	  -H 'Content-Type: application/json' -d '{"model":"q","input":"x"}' | head -c 60; echo "  <- embedding"
	@curl -s -m5 http://127.0.0.1:11437/health; echo "  <- reranker"
	@curl -s -m5 http://127.0.0.1:8000/health; echo "  <- app"

serve:
	uv run uvicorn app.presentation.server:app --port 8000
