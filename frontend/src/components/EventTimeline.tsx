import type { TradeEvent } from "../types";

const LABELS: Record<string, string> = {
  "agent.dispatch": "派发子代理",
  "tool.invoke": "工具开始",
  "tool.result": "工具完成",
  "plan.update": "任务清单",
  "context.compressed": "上下文压缩",
  "model.fallback": "模型回退",
  "cache.hit": "缓存命中",
  "number.unsourced": "金额无出处",
  "final.result": "最终回复",
  error: "异常",
};

/** 截断并如实标注。
 *
 * 不加省略号的硬截断在时间线上看着像 bug 不像设计：`..."total_tokens"` 断在半截、
 * `{"ship_to":null,"t)` 后面还跟着包装用的右括号，读者第一反应是前端挂了。
 * 只有真截到了才加 `…`，没截的短文本不该凭空多个尾巴。
 */
function clip(text: string, max: number): string {
  return text.length > max ? `${text.slice(0, max)}…` : text;
}

/** markdown 源码抹平成单行纯文本。
 *
 * 最终回复本身是 markdown（左侧对话区用 react-markdown 渲染），而时间线是单行摘要，
 * 直接截原文会把 `--- ### ⭐ 首推：` 这种记号露出来——同一屏里左边渲染好、右边露源码。
 */
function plain(text: string): string {
  return text
    .replace(/```[\s\S]*?```/g, " ")      // 代码块整段丢掉，摘要里放不下
    .replace(/^\s*\|.*$/gm, " ")           // 表格整行丢掉：它是结构不是正文
    .replace(/^\s*([-*_]\s*){3,}$/gm, " ") // 分隔线
    .replace(/^\s*#{1,6}\s*/gm, "")        // 标题井号
    .replace(/^\s*>+\s?/gm, "")            // 引用
    .replace(/^\s*[-*+]\s+/gm, "")         // 列表符
    .replace(/[*`]/g, "")                   // 强调记号（不动 `_`：工具名里有下划线）
    .replace(/\s+/g, " ")
    .trim();
}

function summarize(event: TradeEvent): string {
  const p = event.payload ?? {};
  switch (event.type) {
    case "agent.dispatch":
      return `${p.agent}：${clip(String(p.demands ?? ""), 60)}`;
    case "tool.invoke":
      return `${p.tool}（${clip(JSON.stringify(p.args ?? {}), 70)}）`;
    case "tool.result": {
      if (p.circuit) return `${p.tool} 熔断状态 ${p.circuit}：${p.error ?? ""}`;
      // 护栏拒绝与工具自身失败要分开：前者是系统按判据挡下的，后者是外部出了问题，
      // 排查方向完全不同。混成一句"失败"会让人去查工具，而该看的是判据。
      if (p.harness === "rejected") return `${p.tool} 被护栏拒绝：${p.error ?? ""}`;
      if (p.error) return `${p.tool} 失败：${p.error}`;
      if (p.elapsed_ms !== undefined) return `${p.tool}（${p.agent ?? ""}）耗时 ${p.elapsed_ms}ms`;
      if (p.hit_count !== undefined) {
        const strategy = p.recall_strategy ? ` / ${p.recall_strategy}` : "";
        return `${p.tool} 命中 ${p.hit_count} 条${strategy}`;
      }
      if (p.order) return `${p.tool} → ${p.order.order_id} ${p.order.status}`;
      if (p.saved) return `${p.tool} 已记住：${p.saved}`;
      // 计价与组合优化：时间线上原本只显示一个工具名，看不出它到底算出了什么。
      // 到手价是这条链路上最该被看见的数字。
      if (p.covered_need_count !== undefined) {
        const gap = (p.uncovered_needs ?? []).length;
        const remaining =
          p.remaining_major === null || p.remaining_major === undefined
            ? ""
            : ` / 余 ${p.remaining_major}`;
        return `${p.tool} → 到手 ${p.landed_total_major}${remaining}（配齐 ${p.covered_need_count} 项，缺 ${gap} 项）`;
      }
      if (p.landed_total_major !== undefined) {
        return `${p.tool} → 到手 ${p.landed_total_major} ${p.currency ?? ""}`;
      }
      return String(p.tool ?? "");
    }
    case "plan.update":
      return (p.tasks ?? [])
        .map((task: any) => `${task.subject}[${task.state}]`)
        .join(" · ");
    case "context.compressed":
      return `摘要 ${p.summary_length} 字，压缩后上下文 ${p.context_messages} 条`;
    case "model.fallback":
      return `${p.from} 限流，已改用 ${p.to}（${clip(String(p.reason ?? ""), 40)}）`;
    case "cache.hit":
      return `相似度 ${p.similarity}：${clip(String(p.matched_query ?? ""), 40)}`;
    case "number.unsourced": {
      // 告警的价值在于被看见：把具体是哪几个金额、疑似怎么算出来的直接摊在时间线上
      const items = (p.unsourced ?? []) as Array<{ raw: string; kind: string; explain?: string }>;
      const detail = items
        .map((item) => (item.explain ? `${item.raw}（${item.kind}：${item.explain}）` : `${item.raw}（${item.kind}）`))
        .join("、");
      return `${items.length}/${p.total_amounts} 处金额无工具出处：${clip(detail, 90)}`;
    }
    case "final.result":
      return clip(plain(String(p.text ?? "")), 60);
    case "error":
      return String(p.message ?? "");
    default:
      return clip(JSON.stringify(p), 80);
  }
}

export default function EventTimeline({ events }: { events: TradeEvent[] }) {
  return (
    <aside className="timeline">
      <h2>事件时间线</h2>
      {events.length === 0 && <p className="empty">发送一条购物意图后，这里会实时显示 Agent 在做什么。</p>}
      <ol>
        {events.map((event, index) => (
          <li key={index} className={`ev ${event.type.replace(".", "-")}`}>
            <div className="ev-head">
              <span className="tag">{LABELS[event.type] ?? event.type}</span>
              <time>{event.occurred_at.slice(11, 19)}</time>
            </div>
            <div className="ev-body">{summarize(event)}</div>
          </li>
        ))}
      </ol>
    </aside>
  );
}
