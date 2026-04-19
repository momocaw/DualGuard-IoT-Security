import pandas as pd
import numpy as np
import json, time
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, confusion_matrix)

# ── 1. 加载数据（只取前 200 行）────────────────────────────
print("正在读取数据集...")
df_full = pd.read_csv(r"E:\Code\dualguard\data\raw.csv", nrows=200)

# 打印真实列名，方便你确认
print(f"\n数据集列名共 {len(df_full.columns)} 列：")
print(list(df_full.columns))
print(f"\n前 3 行预览：\n{df_full.head(3)}\n")

# ── 2. 自动找标签列和特征列 ───────────────────────────────
# CICIoT2023 标签列通常叫 'label' 或 'Label' 或 'attack'
LABEL_CANDIDATES = ["label", "Label", "attack", "Attack", "class", "Class"]
label_col = None
for c in LABEL_CANDIDATES:
    if c in df_full.columns:
        label_col = c
        break

if label_col is None:
    # 找最后一列作为标签（CICIoT2023 格式）
    label_col = df_full.columns[-1]
    print(f"未找到标准标签列，使用最后一列: '{label_col}'")
else:
    print(f"标签列: '{label_col}'")

print(f"标签值分布:\n{df_full[label_col].value_counts()}\n")

# ── 3. 选数值型特征列（自动排除标签列和非数值列）────────────
numeric_cols = df_full.select_dtypes(include=[np.number]).columns.tolist()
# 排除标签列（如果它是数值的话）
FEATURES = [c for c in numeric_cols if c != label_col]
# 最多取 15 个特征（太多反而影响 IF 效果）
FEATURES = FEATURES[:15]
print(f"使用特征列 ({len(FEATURES)} 个): {FEATURES}\n")

X = df_full[FEATURES].fillna(0).values

# 标签二值化：Benign/BENIGN/Normal = 0，其余 = 1
benign_keywords = ["benign", "BENIGN", "normal", "NORMAL", "0"]
y_true_bin = (~df_full[label_col].astype(str).str.lower()
              .isin(["benign", "normal", "0"])).astype(int).values

print(f"正常流量: {(y_true_bin==0).sum()} 条")
print(f"攻击流量: {(y_true_bin==1).sum()} 条\n")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 4. 孤立森林（无监督）──────────────────────────────────
print("训练孤立森林...")
t0 = time.time()
contamination = max(0.01, min(y_true_bin.mean(), 0.45))  # 自适应污染率
iforest = IsolationForest(n_estimators=100,
                          contamination=float(contamination),
                          random_state=42, n_jobs=-1)
iforest.fit(X_scaled)
if_scores = -iforest.score_samples(X_scaled)
if_preds  = (iforest.predict(X_scaled) == -1).astype(int)
if_latency = (time.time() - t0) * 1000
print(f"IF 完成，耗时 {if_latency:.1f} ms")

# ── 5. 随机森林（有监督，作对比基线）────────────────────────
print("训练随机森林（对比基线）...")
# 样本太少时不拆分，直接全量训练+测试（论文里注明即可）
if len(df_full) >= 20:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y_true_bin, test_size=0.3,
        random_state=42, stratify=y_true_bin if y_true_bin.sum() > 1 else None)
else:
    X_tr, X_te, y_tr, y_te = X_scaled, X_scaled, y_true_bin, y_true_bin

t0 = time.time()
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)
rf_preds_te = rf.predict(X_te)
rf_latency  = (time.time() - t0) * 1000
print(f"RF 完成，耗时 {rf_latency:.1f} ms\n")

# ── 6. 计算并打印指标 ─────────────────────────────────────
def calc_metrics(y, p, name):
    return {
        "method":    name,
        "accuracy":  round(accuracy_score(y, p), 4),
        "precision": round(precision_score(y, p, zero_division=0), 4),
        "recall":    round(recall_score(y, p, zero_division=0), 4),
        "f1":        round(f1_score(y, p, zero_division=0), 4),
    }

m_if = calc_metrics(y_true_bin, if_preds, "Isolation Forest")
m_rf = calc_metrics(y_te, rf_preds_te, "Random Forest")

print("=" * 50)
print("Stage-1 Detection Results (论文 Table I 数据)")
print("=" * 50)
for m in [m_if, m_rf]:
    print(f"\n[{m['method']}]")
    for k, v in m.items():
        if k != "method":
            print(f"  {k:12s}: {v}")
print(f"\nIF 推理延迟 : {if_latency:.1f} ms")
print(f"RF 推理延迟 : {rf_latency:.1f} ms")

# ── 7. 筛出异常行，送给 LLM ──────────────────────────────
df_full["if_score"]   = if_scores
df_full["if_pred"]    = if_preds
df_full["y_true_bin"] = y_true_bin

anomalies = df_full[df_full["if_pred"] == 1].copy()
anomalies["label_col_name"] = label_col  # 记录标签列名给下一步用
print(f"\nIF 标记异常: {len(anomalies)} 条 / {len(df_full)} 条总流量")
print(f"API 调用减少: {(1 - len(anomalies)/len(df_full))*100:.1f}%")

anomalies.to_csv(r"E:\Code\dualguard\data\anomalies_for_llm.csv", index=False)
print("已保存 → data/anomalies_for_llm.csv")

# ── 8. 混淆矩阵图 ────────────────────────────────────────
cm = confusion_matrix(y_true_bin, if_preds)
fig, ax = plt.subplots(figsize=(4, 3.5))
ax.imshow(cm, cmap="Blues")
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(["Normal","Attack"])
ax.set_yticklabels(["Normal","Attack"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Stage-1: Isolation Forest Confusion Matrix")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i,j], ha="center", va="center",
                fontsize=14,
                color="white" if cm[i,j] > cm.max()/2 else "black")
plt.tight_layout()
plt.savefig(r"E:\Code\dualguard\fig_confusion_matrix.png", dpi=150)
plt.show()
print("已保存 → fig_confusion_matrix.png")

# 保存指标
json.dump({
    "iforest": m_if, "rf_supervised": m_rf,
    "if_latency_ms": round(if_latency, 1),
    "rf_latency_ms": round(rf_latency, 1),
    "anomaly_count": int(len(anomalies)),
    "total_count":   int(len(df_full)),
    "label_col":     label_col,
    "features_used": FEATURES
}, open(r"E:\Code\dualguard\data\stage1_metrics.json","w"), indent=2)
print("已保存 → data/stage1_metrics.json")