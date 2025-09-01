import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
)

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric


st.title("Bias Detection App")
st.write("Upload a dataset, choose a target column, and a sensitive attribute to check for bias.")  # [web:23][web:24][web:20]

# ---------------------------
# Helpers
# ---------------------------
def encode_non_sensitive_categoricals(df, target, sensitive):
    """Label-encodes object dtype columns except target and sensitive."""
    df_enc = df.copy()
    for col in df_enc.columns:
        if df_enc[col].dtype == "object" and col not in [target, sensitive]:
            df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))
    return df_enc  # all model features numeric except we keep sensitive separate [web:4][web:11]

def ensure_binary_numeric(series):
    """If target is object/categorical, encode to integers 0/1."""
    if series.dtype == "object":
        return pd.Series(LabelEncoder().fit_transform(series.astype(str)), index=series.index)
    # If numeric with values not in {0,1}, try to map min->0, max->1 (user should ensure binary)
    unique_vals = pd.unique(series.dropna())
    if set(unique_vals) <= {0, 1}:
        return series.astype(int)
    # Fallback: map by sorting unique values (assumes binary)
    if len(unique_vals) == 2:
        vals_sorted = np.sort(unique_vals)
        mapping = {vals_sorted[0]: 0, vals_sorted[1]: 1}
        return series.map(mapping).astype(int)
    return series  # leave as-is for multi-class; AIF360 steps expect binary [web:22][web:14]

def make_numeric_sensitive(series):
    """Encode sensitive attribute to integers for metrics."""
    if series.dtype == "object":
        return pd.Series(LabelEncoder().fit_transform(series.astype(str)), index=series.index)
    return series.astype(int, errors="ignore")


# ---------------------------
# UI: Upload and selections
# ---------------------------
file = st.file_uploader("Upload CSV file", type=["csv"])
if file is not None:
    df = pd.read_csv(file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    target = st.selectbox("Select Target Column (what to predict)", df.columns)
    sensitive = st.selectbox("Select Sensitive Attribute (to check bias)", df.columns)

    if st.button("Check Bias"):
        try:
            # 1) Encode non-sensitive categorical columns
            df_enc = encode_non_sensitive_categoricals(df, target, sensitive)  # [web:4][web:11]

            # 2) Split out X, y and keep sensitive separate from model features
            y = df[target]
            y = ensure_binary_numeric(y)  # ensure numeric/binary for metrics [web:22][web:14]

            s = df[sensitive]
            s_num = make_numeric_sensitive(s)  # numeric vector for metrics [web:22][web:14]

            # X excludes target and sensitive
            X = df_enc.drop(columns=[target, sensitive], errors="ignore")  # avoid strings in X [web:4][web:11]

            # 3) Train/test split (keep sensitive aligned)
            X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
                X, y, s_num, test_size=0.3, random_state=42, stratify=y if len(pd.unique(y)) > 1 else None
            )

            # 4) Train logistic regression
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)  # [web:4]

            # 5) Fairlearn metrics (sensitive passed separately)
            metric_frame = MetricFrame(
                metrics={'accuracy': lambda yt, yp: (yt == yp).mean()},
                y_true=y_test,
                y_pred=y_pred,
                sensitive_features=s_test
            )  # [web:23][web:24][web:20]

            dp_diff = demographic_parity_difference(
                y_test, y_pred, sensitive_features=s_test
            )  # [web:23][web:24][web:20]

            eo_diff = equalized_odds_difference(
                y_test, y_pred, sensitive_features=s_test
            )  # [web:24]

            # 6) AIF360 dataset assembly for binary labels
            # Build a DataFrame including features + label + sensitive
            df_aif = pd.concat(
                [
                    X_test.reset_index(drop=True),
                    pd.Series(y_test, name=target).reset_index(drop=True),
                    pd.Series(s_test, name=sensitive).reset_index(drop=True),
                ],
                axis=1,
            )  # AIF360 expects numeric values in df [web:22]

            # Create BinaryLabelDataset (labels must be binary numeric)
            bld = BinaryLabelDataset(
                favorable_label=1.0,
                unfavorable_label=0.0,
                df=df_aif,
                label_names=[target],
                protected_attribute_names=[sensitive],
            )  # [web:22][web:14]

            # Define groups; by convention treat value 1 as privileged (adjust if needed)
            privileged_groups = [{sensitive: 1}]
            unprivileged_groups = [{sensitive: 0}]

            bld_metric = BinaryLabelDatasetMetric(
                bld,
                privileged_groups=privileged_groups,
                unprivileged_groups=unprivileged_groups,
            )  # [web:14]

            disparate_impact = bld_metric.disparate_impact()  # [web:14]

            # 7) Display results
            st.subheader("Results")
            st.write(f"Model Accuracy: {acc:.3f}")  # [web:4]
            st.write(f"Fairlearn - Accuracy (overall): {metric_frame.overall['accuracy']:.3f}")  # [web:23][web:24]
            st.write(f"Fairlearn - Demographic Parity Difference: {dp_diff:.4f}")  # [web:23][web:24][web:20]
            st.write(f"Fairlearn - Equalized Odds Difference: {eo_diff:.4f}")  # [web:24]
            st.write(f"AIF360 - Disparate Impact: {disparate_impact:.4f}")  # [web:14]

            # Optional: show per-group accuracy
            st.write("Per-group accuracy:")
            st.write(metric_frame.by_group['accuracy'])  # [web:23][web:24]

            # Notes for users
            st.info(
                "Notes: The sensitive attribute is excluded from model training features and only used for fairness metrics. "
                "Ensure the target is binary for AIF360 BinaryLabelDataset; multi-class targets need binarization first."
            )  # [web:22][web:14]

        except Exception as e:
            st.error(f"Error: {e}")  # [web:4]
