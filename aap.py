import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_curve, auc

from fairlearn.metrics import (
    MetricFrame, selection_rate,
    true_positive_rate, false_positive_rate, false_negative_rate, count,
    demographic_parity_difference, equalized_odds_difference,
)
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric


st.title("Bias Detection App")
st.write("Upload a dataset, choose a target column, and a sensitive attribute to check for bias (with decoded labels, visuals, and a natural‑language summary).")  # [web:148][web:25]

# ---------------------------
# Helpers
# ---------------------------
def encode_non_sensitive_categoricals(df, target, sensitive):
    df_enc = df.copy()
    encoders = {}
    for col in df_enc.columns:
        if df_enc[col].dtype == "object" and col not in [target, sensitive]:
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
            encoders[col] = le
    return df_enc, encoders  # [web:148]

def ensure_binary_numeric(series):
    if series.dtype == "object":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(series.astype(str)), index=series.index)
        return y, le
    unique_vals = pd.unique(series.dropna())
    if set(unique_vals) <= {0, 1}:
        return series.astype(int), None
    if len(unique_vals) == 2:
        vals_sorted = np.sort(unique_vals)
        mapping = {vals_sorted[0]: 0, vals_sorted[1]: 1}
        y = series.map(mapping).astype(int)
        le = LabelEncoder()
        le.classes_ = np.array(vals_sorted)
        return y, le
    return series, None  # multi-class not supported by ROC/AIF360 [web:25]

def fit_sensitive_encoder(series):
    if series.dtype == "object":
        le = LabelEncoder()
        s_num = pd.Series(le.fit_transform(series.astype(str)), index=series.index)
        return s_num, le
    le = LabelEncoder()
    le.classes_ = np.array(sorted(pd.unique(series.dropna())))
    s_num = series.astype(int)
    return s_num, le  # [web:148]

def decode_groups(index_like, sensitive_le):
    try:
        decoded = pd.Index(sensitive_le.inverse_transform(pd.Series(index_like).astype(int)))
    except Exception:
        decoded = pd.Index(pd.Series(index_like).astype(str))
    return decoded  # [web:148]

def rotate_align_xticks(ax, rotation=45, ha='right'):
    plt.setp(ax.get_xticklabels(), rotation=rotation, ha=ha, rotation_mode='anchor')  # [web:124][web:126]

def plot_metricframe_bars(mf: MetricFrame, title="Per-group metrics"):
    data = mf.by_group.copy()
    metric_cols = [c for c in data.columns if c != "count"]
    if not metric_cols:
        return
    rows = 2
    cols = int(np.ceil(len(metric_cols) / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(metric_cols):
        ax = axes[i]
        data[col].plot(kind='bar', ylim=[0, 1], ax=ax, title=col)
        rotate_align_xticks(ax, rotation=45, ha='right')
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle(title)
    plt.tight_layout()
    st.pyplot(fig)  # [web:148]

def plot_group_counts(mf: MetricFrame):
    if "count" in mf.by_group.columns:
        fig, ax = plt.subplots(figsize=(6, 3))
        mf.by_group["count"].plot(kind='bar', ax=ax, title="Group sample sizes")
        rotate_align_xticks(ax, rotation=45, ha='right')
        st.pyplot(fig)  # [web:148]

def plot_roc_by_group(y_true, y_score, s_test, group_labels):
    groups = pd.unique(s_test)
    fig, ax = plt.subplots(figsize=(6, 5))
    for g in groups:
        idx = (s_test == g)
        if np.sum(y_true[idx] == 1) == 0 or np.sum(y_true[idx] == 0) == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true[idx], np.array(y_score)[idx])
        ax.plot(fpr, tpr, label=f"{group_labels[g]} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC by group")
    ax.legend()
    st.pyplot(fig)  # [web:25]

def summarize_results(acc, dp_diff, eo_diff, di, is_binary, sensitive_name):
    # Heuristic thresholds (tune per domain)
    acc_quality = "strong" if acc >= 0.90 else ("good" if acc >= 0.80 else "modest")
    dp_note = "near parity" if abs(dp_diff) < 0.02 else ("moderate gap" if abs(dp_diff) < 0.10 else "large gap")
    eo_note = "small gap" if abs(eo_diff) < 0.05 else ("moderate gap" if abs(eo_diff) < 0.15 else "large gap")
    di_note = None
    if di is not None:
        di_note = "near parity" if 0.8 <= di <= 1.25 else ("potential concern" if 0.7 <= di < 0.8 or 1.25 < di <= 1.4 else "high concern")

    lines = []
    lines.append(f"Accuracy is {acc:.3f}, which is {acc_quality} for many binary tasks; pair utility with fairness metrics to assess equity.")  # [web:148]
    lines.append(f"Demographic Parity Difference = {dp_diff:.4f} → {dp_note}; this compares selection rates across {sensitive_name} groups and 0 indicates equal allocation.")  # [web:25]
    lines.append(f"Equalized Odds Difference = {eo_diff:.4f} → {eo_note}; this compares TPR and FPR across groups and 0 indicates similar error behaviour.")  # [web:25]
    if is_binary and di is not None:
        lines.append(f"Disparate Impact = {di:.4f} → {di_note}; values close to 1 indicate balanced positive outcome rates between unprivileged and privileged groups.")  # [web:37]
    else:
        lines.append("Disparate Impact not reported for non-binary targets; it is defined for binary outcomes only.")  # [web:25]

    bias_msg = "No strong evidence of bias on these metrics; allocation and error‑rate gaps appear small." if (abs(dp_diff) < 0.02 and abs(eo_diff) < 0.05 and (di is None or (0.8 <= di <= 1.25))) else "Some disparities are indicated; inspect per‑group TPR/FPR and selection rates to locate the drivers."
    lines.append(bias_msg)  # [web:148][web:25]

    return " ".join(lines)

# ---------------------------
# UI
# ---------------------------
file = st.file_uploader("Upload CSV file", type=["csv"])
if file is not None:
    df = pd.read_csv(file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    target = st.selectbox("Select Target Column (what to predict)", df.columns)
    sensitive = st.selectbox("Select Sensitive Attribute (to check bias)", df.columns)

    show_plots = st.multiselect(
        "Visualizations",
        ["Per-group bars", "Group counts", "ROC by group"],
        default=["Per-group bars", "Group counts", "ROC by group"]
    )

    if st.button("Check Bias"):
        try:
            # 1) Encode non-sensitive categoricals
            df_enc, _ = encode_non_sensitive_categoricals(df, target, sensitive)

            # 2) Prepare y and sensitive
            y_raw = df[target]
            y, y_le = ensure_binary_numeric(y_raw)
            s_raw = df[sensitive]
            s_num, s_le = fit_sensitive_encoder(s_raw)

            # X excludes target and sensitive
            X = df_enc.drop(columns=[target, sensitive], errors="ignore")

            # 3) Split
            X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
                X, y, s_num, test_size=0.3, random_state=42, stratify=y if len(pd.unique(y)) > 1 else None
            )

            # 4) Model
            model = LogisticRegression(max_iter=2000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            # Scores for ROC
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_score = model.decision_function(X_test)
            else:
                y_score = y_pred

            # Binary guard (needed for ROC & AIF360)
            unique_y = set(pd.unique(y_test))
            is_binary = unique_y <= {0, 1} or unique_y <= {-1, 1}  # [web:132]

            # 5) Metrics
            metrics = {
                'accuracy': lambda yt, yp: (yt == yp).mean(),
                'selection_rate': selection_rate,
                'TPR': true_positive_rate,
                'FPR': false_positive_rate,
                'FNR': false_negative_rate,
                'count': count,
            }
            mf = MetricFrame(metrics=metrics, y_true=y_test, y_pred=y_pred, sensitive_features=s_test)
            dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=s_test)  # [web:25]
            eo_diff = equalized_odds_difference(y_test, y_pred, sensitive_features=s_test)      # [web:25]

            # Decode labels for display
            decoded_index = decode_groups(mf.by_group.index, s_le)
            mf.by_group.index = decoded_index
            group_label_map = {code: name for code, name in zip(range(len(s_le.classes_)), s_le.classes_)}

            # 6) AIF360 Disparate Impact (only binary)
            if is_binary:
                df_aif = pd.concat(
                    [
                        X_test.reset_index(drop=True),
                        pd.Series(y_test, name=target).reset_index(drop=True),
                        pd.Series(s_test, name=sensitive).reset_index(drop=True),
                    ],
                    axis=1,
                )
                bld = BinaryLabelDataset(
                    favorable_label=1.0,
                    unfavorable_label=0.0,
                    df=df_aif,
                    label_names=[target],
                    protected_attribute_names=[sensitive],
                )
                privileged_groups = [{sensitive: 1}]
                unprivileged_groups = [{sensitive: 0}]
                bld_metric = BinaryLabelDatasetMetric(bld, privileged_groups, unprivileged_groups)
                disparate_impact = bld_metric.disparate_impact()  # [web:9]
            else:
                disparate_impact = None  # [web:25]

            # 7) Display numeric results
            st.subheader("Results")
            st.write(f"Model Accuracy: {acc:.3f}")  # [web:148]
            st.write(f"Fairlearn - Demographic Parity Difference: {dp_diff:.4f}")  # [web:25]
            st.write(f"Fairlearn - Equalized Odds Difference: {eo_diff:.4f}")  # [web:25]
            if disparate_impact is not None:
                st.write(f"AIF360 - Disparate Impact: {disparate_impact:.4f}")  # [web:37]
            else:
                st.warning("Disparate Impact skipped: target is not binary. Select a binary target to compute this metric.")  # [web:25]

            # 8) Add natural‑language summary
            st.subheader("Interpretation")
            summary_text = summarize_results(acc, dp_diff, eo_diff, disparate_impact, is_binary, sensitive)
            st.write(summary_text)  # [web:148][web:25][web:37]

            # 9) Visualizations
            if "Per-group bars" in show_plots:
                plot_metricframe_bars(mf, title="Per-group metrics")  # [web:148]
            if "Group counts" in show_plots:
                plot_group_counts(mf)  # [web:148]
            if "ROC by group" in show_plots:
                if is_binary and hasattr(model, "predict_proba"):
                    plot_roc_by_group(y_test.values, y_score, s_test.values, group_label_map)  # [web:25]
                else:
                    st.warning("ROC by group skipped: target must be binary (labels in {0,1} or {-1,1}).")  # [web:132]

            st.info(
                "Metric meanings: Demographic parity checks selection-rate parity across groups; "
                "Equalized odds checks parity of TPR and FPR; Disparate impact is the ratio of selection rates (unprivileged/privileged). "
                "Choose thresholds appropriate to the domain; small gaps may still matter in high‑stakes uses."  # [web:25][web:35][web:37]
            )

        except Exception as e:
            st.error(f"Error: {e}")
