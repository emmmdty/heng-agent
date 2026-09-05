# -*- coding: utf-8 -*-
"""harness

Agent 运行时护栏的判定层（纯逻辑，不碰 IO）：

    loop_detector      第 4 项  滑动窗口循环检测——同一工具反复横跳时注入收敛提示
    assertions         第 2 项  三类单步断言（Schema / Sequencing / Semantic）
    drift_detector     第 3 项  Silent-Drift 静默漂移检测
    number_provenance  八期    回复里的金额必须有工具出处
    order_provenance   十四期  下单的商品必须有工具出处（写路径，可硬拒）
    contact_provenance 二十一期 回复里的收货地址/电话/邮编必须有出处（不许编造买家信息）
    run_identity       十期    这一轮读数是哪套配置、哪份代码跑出来的

这些判定由 app/infrastructure/harness_middleware.py 在工具边界调用，
由 app/application/agents/orchestrator.py 在轮次边界调用。
判定层不依赖 AgentScope，可单独单测。
"""
