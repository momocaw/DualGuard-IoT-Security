import json, numpy as np
import matplotlib.pyplot as plt
from collections import Counter

BASE = r"E:\Code\dualguard"

s1  = json.load(open(f"{BASE}\\data\\stage1_metrics.json"))
s2  = json.load(open(f"{BASE}\\data\\stage2_metrics.json"))
llm = json.load(open(f"{BASE}\\data\\llm_results.json"))

if_m  = s1["iforest"]
rf_m  = s1["rf_supervised"]
if_lat   = s1["if_latency_ms"]
llm_lat  = s2["avg_llm_latency_ms"]
n_total  = s1["total_count"]
n_anom   = s1["anomaly_count"]

# ── 打印论文数据表 ────────────────────────────────────────
print("\n" + "="*56)
print("TABLE I — Detection Performance (论文直接用这组数字)")
print("="*56)
print(f"{'Method':<28} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
print("-"*56)
for m in [if_m, rf_m]:
    print(f"{m['method']:<28} {m['accuracy']:>6.4f} "
          f"{m['precision']:>6.4f} {m['recall']:>6.4f} {m['f1']:>6.4f}")

print("\n" + "="*56)
print("TABLE II — Latency & API Efficiency (论文直接用这组数字)")
print("="*56)
reduction = (1 - n_anom/n_total)*100
print(f"  总流量               : {n_total} 条")
print(f"  IF 推理延迟           : {if_lat:.1f} ms")
print(f"  IF 标记异常数         : {n_anom} 条")
print(f"  LLM 平均延迟          : {llm_lat:.1f} ms/条")
print(f"  API 调用减少          : {reduction:.1f}%  ← 论文核心数据")
print(f"  成功生成报告          : {s2['total_reports']} 条")

attack_ct  = Counter(r["attack_type"] for r in llm)
severity_ct = Counter(r["severity"]   for r in llm)

print("\n" + "="*56)
print("TABLE III — LLM Attack Type Distribution")
print("="*56)
for k, v in sorted(attack_ct.items(), key=lambda x: -x[1]):
    print(f"  {k:<20}: {v:>3} ({v/len(llm)*100:.1f}%)")

# ── 生成三张图 ────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# 图1: 性能对比
names = ["Accuracy","Precision","Recall","F1"]
iv = [if_m["accuracy"], if_m["precision"], if_m["recall"], if_m["f1"]]
rv = [rf_m["accuracy"], rf_m["precision"], rf_m["recall"], rf_m["f1"]]
x = np.arange(len(names)); w = 0.35
axes[0].bar(x-w/2, iv, w, label="Isolation Forest", color="#B5D4F4", edgecolor="white")
axes[0].bar(x+w/2, rv, w, label="Random Forest",    color="#185FA5", edgecolor="white")
axes[0].set_xticks(x); axes[0].set_xticklabels(names)
axes[0].set_ylim(0, 1.1); axes[0].set_title("(a) Stage-1 Performance")
axes[0].legend(fontsize=9); axes[0].set_ylabel("Score")
for i,(a,b) in enumerate(zip(iv,rv)):
    axes[0].text(i-w/2, a+0.02, f"{a:.2f}", ha="center", fontsize=8)
    axes[0].text(i+w/2, b+0.02, f"{b:.2f}", ha="center", fontsize=8)

# 图2: API 调用减少
axes[1].bar(["Naive\n(LLM-only)", "DualGuard\n(Proposed)"],
            [n_total, n_anom], color=["#F09595","#9FE1CB"],
            width=0.4, edgecolor="white")
axes[1].set_ylabel("LLM API calls")
axes[1].set_title("(b) API Call Reduction")
for i,(x_,v) in enumerate(zip([0,1],[n_total,n_anom])):
    axes[1].text(i, v+0.5, str(v), ha="center", fontsize=12, fontweight="bold")
axes[1].annotate(f"↓{reduction:.0f}%",
                 xy=(1, n_anom), xytext=(0.5, n_total*0.5),
                 arrowprops=dict(arrowstyle="->",color="#A32D2D"),
                 fontsize=11, color="#A32D2D", ha="center")

# 图3: 严重等级饼图
if severity_ct:
    sev_colors = {"Critical":"#F09595","High":"#FAC775",
                  "Medium":"#9FE1CB","Low":"#B5D4F4","Unknown":"#D3D1C7"}
    labels = list(severity_ct.keys())
    vals   = list(severity_ct.values())
    cols   = [sev_colors.get(l,"#D3D1C7") for l in labels]
    axes[2].pie(vals, labels=labels, colors=cols,
                autopct="%1.1f%%", startangle=90)
    axes[2].set_title("(c) Severity Distribution")
else:
    axes[2].text(0.5, 0.5, "No data", ha="center", va="center")
    axes[2].set_title("(c) Severity Distribution")

plt.tight_layout()
plt.savefig(f"{BASE}\\fig_results_combined.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"\n图表已保存 → fig_results_combined.png")

# ── 打印一条示例报告（放进论文正文）─────────────────────────
if llm:
    eg = llm[0]
    print("\n" + "="*56)
    print("示例 LLM 报告（放进论文 Section IV Case Study）")
    print("="*56)
    print(f"  真实标签    : {eg['true_label']}")
    print(f"  IF异常分数  : {eg['if_score']}")
    print(f"  LLM判断类型 : {eg['attack_type']}")
    print(f"  严重等级    : {eg['severity']}")
    print(f"  触发指标    : {eg['key_indicators']}")
    print(f"  建议动作    : {eg['recommendation']}")