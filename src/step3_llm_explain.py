from config import get_client, MODEL
import pandas as pd
import json, time

client = get_client()

# ── 加载 Stage-1 筛出的异常样本 ──────────────────────────
df = pd.read_csv(r"E:\Code\dualguard\data\anomalies_for_llm.csv")
s1 = json.load(open(r"E:\Code\dualguard\data\stage1_metrics.json"))
label_col = s1.get("label_col", "label")
features  = s1.get("features_used", [])

# 只处理前 30 条，控制时间和费用
SAMPLE_N = min(30, len(df))
df_sample = df.sample(n=SAMPLE_N, random_state=42).reset_index(drop=True)
print(f"将对 {SAMPLE_N} 条异常流量调用 LLM 生成解释报告...\n")

PROMPT_TEMPLATE = """You are a cybersecurity analyst for an IoT network.
The following network flow was flagged as anomalous by an Isolation Forest model.

Flow statistics:
{features}

Anomaly score: {score:.3f} (higher = more anomalous)
Ground-truth label: {true_label}

Provide a security alert report in JSON format only.
Return ONLY valid JSON, no markdown, no explanation outside the JSON.
Required keys:
  "attack_type": one of [DDoS, DoS, PortScan, BruteForce, Malware, Benign_FP, Unknown]
  "severity": one of [Critical, High, Medium, Low]
  "key_indicators": list of 2-3 feature names that triggered the alert
  "recommendation": one sentence action for the network operator
"""

results  = []
latencies = []

for i, row in df_sample.iterrows():
    # 构造特征描述字符串
    feat_lines = []
    for f in features:
        if f in row and f not in ("if_score","if_pred","y_true_bin","label_col_name"):
            val = row[f]
            if isinstance(val, float):
                feat_lines.append(f"  {f}: {val:.4f}")
            else:
                feat_lines.append(f"  {f}: {val}")
    feat_str = "\n".join(feat_lines)

    true_label = row.get(label_col, "Unknown")
    score      = float(row.get("if_score", 0.5))

    prompt = PROMPT_TEMPLATE.format(
        features=feat_str,
        score=score,
        true_label=true_label
    )

    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300,
            # ★ 关键：关闭 Qwen3 的思考模式，否则输出会带 <think>...</think>
            extra_body={"enable_thinking": False}
        )
        raw = resp.choices[0].message.content.strip()
        lat = (time.time() - t0) * 1000
        latencies.append(lat)

        # 清理可能残留的 <think> 块和 markdown 代码块
        import re
        raw_clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        raw_clean = raw_clean.replace("```json","").replace("```","").strip()

        # 找到第一个 { 到最后一个 } 之间的内容
        start = raw_clean.find("{")
        end   = raw_clean.rfind("}") + 1
        if start != -1 and end > start:
            raw_clean = raw_clean[start:end]

        parsed = json.loads(raw_clean)

        results.append({
            "row_index":      int(i),
            "true_label":     str(true_label),
            "if_score":       round(score, 3),
            "attack_type":    parsed.get("attack_type", "Unknown"),
            "severity":       parsed.get("severity", "Unknown"),
            "key_indicators": parsed.get("key_indicators", []),
            "recommendation": parsed.get("recommendation", ""),
            "latency_ms":     round(lat, 1),
        })
        print(f"[{i+1:2d}/{SAMPLE_N}] {str(true_label):<20} → "
              f"{parsed.get('attack_type','?'):<12} "
              f"({parsed.get('severity','?'):<8}) | {lat:.0f}ms")

    except json.JSONDecodeError as e:
        print(f"[{i+1:2d}/{SAMPLE_N}] JSON解析失败: {e}")
        print(f"  原始输出: {raw[:200]}")
    except Exception as e:
        print(f"[{i+1:2d}/{SAMPLE_N}] 请求失败: {e}")

    time.sleep(0.5)  # 避免触发限流

# ── 保存结果 ─────────────────────────────────────────────
with open(r"E:\Code\dualguard\data\llm_results.json","w",encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

avg_lat = sum(latencies)/len(latencies) if latencies else 0
print(f"\n完成！成功生成 {len(results)} 条报告")
print(f"LLM 平均延迟: {avg_lat:.1f} ms/条")
print(f"失败条数: {SAMPLE_N - len(results)}")

json.dump({"avg_llm_latency_ms": round(avg_lat,1),
           "total_reports": len(results),
           "failed": SAMPLE_N - len(results)},
          open(r"E:\Code\dualguard\data\stage2_metrics.json","w"), indent=2)
print("已保存 → data/stage2_metrics.json 和 data/llm_results.json")