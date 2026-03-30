import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
)
from evaluation import (
    load_models, prepare_dataset,
    evaluate_xgb, evaluate_iso, evaluate_ensemble,
    hybrid_score, risk_label, calculate_cost_metrics, tune_hybrid_weights,
    tune_threshold_by_cost,full_evaluation_report
)
import os, sys
import shap

# === PATHS ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)  # one level up (project root)
MODEL_DIR = os.path.join(ROOT_DIR, "models")
DATA_DIR = os.path.join(ROOT_DIR, "data")

FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_columns.joblib")
ISO_FEATURES_PATH = os.path.join(MODEL_DIR, "iso_training_columns.joblib")
ISO_SCALER_PATH = os.path.join(MODEL_DIR, "iso_scaler.joblib")
ISO_NUMERIC_FEATURES_PATH = os.path.join(MODEL_DIR, "iso_numeric_features.joblib")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.joblib")
ISO_MODEL_PATH = os.path.join(MODEL_DIR, "iso_model.joblib")
ENSEMBLE_BASE_PATH= os.path.join(MODEL_DIR, "ensemble_base_models.joblib")
ENSEMBLE_META_PATH= os.path.join(MODEL_DIR, "ensemble_meta_model.joblib")
MASTER_DATA_PATH = os.path.join(DATA_DIR, "master_dataset.parquet")
sys.path.append(BASE_DIR)

# === Streamlit config ===
st.set_page_config(page_title="💳 Fraud Detection Dashboard", layout="wide", initial_sidebar_state="expanded")
sns.set_theme(style="whitegrid")  # consistent plot style

# === Load models ===
@st.cache_resource
def load_all_models():
    xgb_model = joblib.load(XGB_MODEL_PATH)
    iso_model = joblib.load(ISO_MODEL_PATH)
    iso_scaler = joblib.load(ISO_SCALER_PATH)
    iso_features = joblib.load(ISO_FEATURES_PATH)
    iso_numeric_features = joblib.load(ISO_NUMERIC_FEATURES_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    ensemble_base = joblib.load(ENSEMBLE_BASE_PATH)
    ensemble_meta = joblib.load(ENSEMBLE_META_PATH)
    return xgb_model, iso_model, iso_scaler, iso_features, iso_numeric_features, feature_cols, ensemble_base, ensemble_meta
xgb_model, iso_model, iso_scaler, iso_features, iso_numeric_features, feature_columns, ensemble_base, ensemble_meta = load_all_models()
st.sidebar.success("✅ Models loaded successfully!")

# === Load data ===
@st.cache_data
def load_data():
    test_path = os.path.join(DATA_DIR, "test_data.parquet")
    if os.path.exists(test_path):
        df = pd.read_parquet(test_path)
        if "Class" not in df.columns:
            st.warning("No fraud labels found — running in prediction-only mode.")
        return df
    else:
        st.warning("Test data not found. Please upload manually.")
        return None

data = load_data()

# === Sidebar Controls ===s
st.sidebar.markdown("### 🧠 Model Details")
st.sidebar.write(f"XGBoost: {len(xgb_model.feature_importances_)} features")
st.sidebar.write(f"Hybrid: XGBoost + IsolationForest features ({len(xgb_model.feature_importances_)} + {len(iso_features)})")
st.sidebar.write(f"Ensemble: {len(feature_columns)} features (base models + meta)")  # assuming ensemble meta uses same features
with st.sidebar.expander("⚙️ Model Configuration", expanded=True):
    # Model selection
    model_choice = st.radio(
        "Select model",
        ["XGBoost", "IsolationForest", "Hybrid", "Ensemble"]
    )
top_n_frauds = st.sidebar.slider("Top N transactions by fraud score", 5, 500, 20)
top_n_cost = st.sidebar.slider("Top N transactions by cost (Hybrid only)", 10, 500, 50)

if data is not None:
    X, y = prepare_dataset(data)
    numeric_features = iso_numeric_features

# === Auto / Manual Hybrid Weights ===
if model_choice == "Hybrid" and data is not None:
    st.sidebar.markdown("### ⚙️ Hybrid Weights")
    auto_tune = st.sidebar.checkbox("Auto-tune weights", value=False)  # 🔧 FIX: default off
    if auto_tune and st.sidebar.button("Run weight tuning"):  # 🔧 FIX: explicit button
        with st.spinner("Tuning hybrid weights... please wait ⏳"):
            best_wxgb, best_wiso = tune_hybrid_weights(
                y_true=y, model_type='xgb', X=X,
                xgb_model=xgb_model, iso_model=iso_model,
                iso_score_scaler=iso_scaler,
                numeric_features=iso_numeric_features,
                iso_features=iso_features
            )
        w_xgb, w_iso = best_wxgb, best_wiso
        st.sidebar.success(f"Optimal Weights → XGBoost: {w_xgb:.2f}, IsolationForest: {w_iso:.2f}")
    else:
        w_xgb = st.sidebar.slider("XGBoost Weight", 0.0, 1.0, 0.7, 0.01)
        w_iso = 1 - w_xgb
else:
    w_xgb, w_iso = 0.7, 0.3

# === Unified scoring function ===
@st.cache_data(show_spinner=False, max_entries=3)
def compute_scores(X, y, model_choice, numeric_features, w_xgb=0.7, w_iso=0.3):  # 🔧 FIX: added y and numeric_features
    if model_choice == "XGBoost":
        scores, metrics = evaluate_xgb(xgb_model, X, y)
    elif model_choice == "IsolationForest":
        scores, metrics = evaluate_iso(iso_model, iso_scaler, X, y, numeric_features, iso_features)
    elif model_choice == "Hybrid":
        scores, _, _ = hybrid_score(
            xgb_model, iso_model, X,
            w_xgb=w_xgb, w_iso=w_iso,
            iso_score_scaler=iso_scaler,
            numeric_features=iso_numeric_features,
            iso_features=iso_features,
            model_type='xgb',
            ensemble_base=None,
            ensemble_meta=None
        )
        metrics = full_evaluation_report(y, scores)
    elif model_choice == "Ensemble":
        scores, metrics = evaluate_ensemble(ensemble_base, ensemble_meta, X, y)
    else:
        raise ValueError(f"Unknown model_choice: {model_choice}")
    return scores, metrics

# Compute scores
if data is not None:
    scores, _ = compute_scores(X, y, model_choice, numeric_features, w_xgb, w_iso)  # 🔧 FIX: updated call
    df_processed = data.copy()
    df_processed["fraud_score"] = scores
    # FIX: toggle for auto vs manual threshold
    use_auto_thresh = st.sidebar.checkbox("Auto-optimize threshold by cost", value=True)
    if use_auto_thresh:
        threshold_info = tune_threshold_by_cost(y, scores, fp_cost=20, avg_fraud_amount=2000)
        threshold = threshold_info["threshold"]
        st.sidebar.write(f"Optimal threshold (by cost): {threshold:.3f}")
    else:
        threshold = st.sidebar.slider("Manual Threshold", 0.0, 1.0, 0.5, 0.01)

    df_processed["pred_label"] = (scores >= threshold).astype(int)

    # 🔧 FIX: Defensive risk_label assignment before using df_alerts or display
    if "hybrid_score" in df_processed.columns:
        score_col_for_label = "hybrid_score"
    elif "fraud_score" in df_processed.columns:
        score_col_for_label = "fraud_score"
    elif "score" in df_processed.columns:
        score_col_for_label = "score"
    else:
        score_col_for_label = None

    if score_col_for_label:
        df_processed["risk_label"] = df_processed[score_col_for_label].apply(risk_label)
    else:
        df_processed["risk_label"] = "Unknown"
else:
    st.warning("No data available for scoring.")
    st.stop()

def compute_single_score(sample_df, model_choice, numeric_features, w_xgb=0.5, w_iso=0.5):
    """Wrapper for one-transaction scoring (returns scalar)."""
    y_dummy = np.zeros(len(sample_df))  # dummy since y not used for prediction
    scores, _ = compute_scores(sample_df, y_dummy, model_choice, numeric_features, w_xgb, w_iso)
    return float(scores[0]) if isinstance(scores, (np.ndarray, list, pd.Series)) else float(scores)

# ---------------------------
# Evaluation Metrics
# ---------------------------
if y is not None:
    metrics = full_evaluation_report(y, scores)
    st.subheader(f"{model_choice} Metrics")
    metric_cols = st.columns(5)
    metric_cols[0].metric("Accuracy", f"{metrics['Accuracy']:.3f}")
    metric_cols[1].metric("Precision", f"{metrics['Precision']:.3f}")
    metric_cols[2].metric("Recall", f"{metrics['Recall']:.3f}")
    metric_cols[3].metric("F1-Score", f"{metrics['F1-Score']:.3f}")
    metric_cols[4].metric("ROC-AUC", f"{metrics['ROC-AUC']:.3f}")

    # Classification Report
    st.subheader("Detailed Classification Report")
    report = classification_report(y, (scores >= threshold).astype(int), output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.dataframe(df_report.style.format("{:.3f}"), use_container_width=True)

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = metrics["ConfusionMatrix"]
    fig, ax = plt.subplots()
    sns.heatmap(
        [[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]],
        annot=True, fmt="d", cmap="Blues",
        xticklabels=["Legit","Fraud"], yticklabels=["Legit","Fraud"], ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # ROC Curve
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y, scores)
    roc_auc = metrics["ROC-AUC"]
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    st.pyplot(fig)

    # Precision-Recall Curve
    st.subheader("Precision-Recall Curve")
    precision_vals, recall_vals, _ = precision_recall_curve(y, scores)
    fig, ax = plt.subplots()
    ax.plot(recall_vals, precision_vals, color="purple", lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    st.pyplot(fig)

# --------------------
# Calculate risk label distribution (only if available) (for Hybrid & Ensemble)
# --------------------
if model_choice in ["Hybrid", "Ensemble"]:
    # Convert 0–1 → 0–100 for visualization / risk labeling
    hybrid_0_100 = (scores * 100).astype(float)
    # Apply risk label (Green / Yellow / Red)
    risk_labels_array = np.vectorize(risk_label)(hybrid_0_100)
    df_processed["risk_label"] = risk_labels_array

    st.subheader(f"{model_choice} Risk Label Distribution")
    risk_counts = pd.Series(risk_labels_array).value_counts(normalize=True) * 100
    st.bar_chart(risk_counts)

if "risk_label" in df_processed.columns:
    risk_counts = df_processed['risk_label'].value_counts(normalize=True) * 100
    print("[INFO] Percentage of transactions by risk label:")
    for label in ['Green', 'Yellow', 'Red']:
        pct = risk_counts.get(label, 0.0)
        print(f"  {label}: {pct:.2f}%")

    # ---- Hybrid score histogram ----
    plt.figure(figsize=(8,5))
    plt.hist(hybrid_0_100, bins=50, color='skyblue', edgecolor='k')
    plt.xlabel("Hybrid Risk Score (0-100)")
    plt.ylabel("Transaction Count")
    plt.title(f"{model_choice} Risk Score Distribution")
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt.gcf())

    # ---- Pie chart of risk labels ----
    plt.figure(figsize=(6,6))
    colors = {"Green":"green", "Yellow":"gold", "Red":"red"}
    plt.pie(
        risk_counts.reindex(['Green','Yellow','Red']).fillna(0),
        labels=['Green','Yellow','Red'],
        autopct='%1.1f%%',
        startangle=90,
        colors=[colors[label] for label in ['Green','Yellow','Red']],
        wedgeprops={"edgecolor":"black"}
    )
    plt.title(f"{model_choice} Transactions by Risk Label")
    plt.tight_layout()
    st.pyplot(plt.gcf())


# -------------------------------
# Alerts DataFrame
# -------------------------------
def generate_alerts(df, hybrid_scores, p_fraud, iso_scaled, threshold=0.5,
                    avg_fraud_amount=2000, fp_cost=20):
    df_alerts = df.copy()
    df_alerts["hybrid_score"] = hybrid_scores
    df_alerts["xgb_prob"] = p_fraud
    df_alerts["iso_score"] = iso_scaled
    df_alerts["alert"] = (hybrid_scores >= threshold).astype(int)
    
    if "Amount" in df_alerts.columns:
        amounts = df_alerts["Amount"].values
    else:
        amounts = np.ones(len(df_alerts)) * avg_fraud_amount
    
    df_alerts["missed_fraud_cost"] = np.where(
        (df_alerts["alert"] == 0) & (df_alerts["Class"] == 1),
        amounts, 0
    )
    df_alerts["false_alert_cost"] = np.where(
        (df_alerts["alert"] == 1) & (df_alerts["Class"] == 0),
        fp_cost, 0
    )
    df_alerts["total_cost"] = df_alerts["missed_fraud_cost"] + df_alerts["false_alert_cost"]
    alerts_only = df_alerts[df_alerts["alert"] == 1].copy()
    return df_alerts, alerts_only

# === Generate Alerts DataFrame (for Cost & Heatmap) ===
df_alerts, alerts_only = generate_alerts(
    df_processed,
    hybrid_scores=scores,
    p_fraud=scores,        # placeholder (same as scores for now)
    iso_scaled=scores,     # placeholder (same as scores)
    threshold=threshold,
    avg_fraud_amount=2000,
    fp_cost=20
)

# ==========================
# Display Alerts Table
# ==========================
# === Risk Label Assignment ===
if "hybrid_score" in df_alerts.columns:
    df_alerts["risk_label"] = df_alerts["hybrid_score"].apply(risk_label)
elif "score" in df_alerts.columns:
    df_alerts["risk_label"] = df_alerts["score"].apply(risk_label)
else:
    st.warning("⚠️ No score column found to derive risk labels.")

st.subheader("🚨 Alerts Table")
tab1, tab2 = st.tabs(["All Transactions", "Flagged Alerts Only"])
with tab1:
    st.dataframe(
        df_alerts.sort_values("hybrid_score", ascending=False)
        [["hybrid_score", "alert", "risk_label", "total_cost", "missed_fraud_cost", "false_alert_cost"] + 
         ([ "Class"] if "Class" in df_alerts.columns else [])]
        .reset_index(drop=True),
        use_container_width=True
    )
with tab2:
    if len(alerts_only) > 0:
        st.dataframe(
            alerts_only.sort_values("hybrid_score", ascending=False)
            [["hybrid_score", "risk_label", "total_cost", "missed_fraud_cost", "false_alert_cost"] + 
             ([ "Class"] if "Class" in alerts_only.columns else [])]
            .reset_index(drop=True),
            use_container_width=True
        )
    else:
        st.info("No transactions triggered alerts at the current threshold.")

# -------------------------------
# Plotting
# -------------------------------
def plot_top_by_region(df, region_col='location', score_col='hybrid_score', top_n=10, savepath=None):
    agg = df.groupby(region_col)[score_col].mean().nlargest(top_n)
    fig, ax = plt.subplots(figsize=(8,4))
    agg.plot(kind='bar', ax=ax)
    ax.set_ylabel('Average Hybrid Risk')
    plt.tight_layout()
    if savepath: plt.savefig(savepath)
    return fig

def plot_cost_heatmap(alerts_df, top_n=200, figsize=(12,6)):
    df_top = alerts_df.nlargest(top_n, "total_cost")
    cost_cols = ["missed_fraud_cost", "false_alert_cost", "total_cost"]
    df_cost = df_top[cost_cols].T
    plt.figure(figsize=figsize)
    sns.heatmap(df_cost, annot=True, fmt=".0f")
    plt.title(f"Cost Heatmap (top {top_n} transactions by total_cost)")
    plt.xlabel("Transactions")
    plt.ylabel("Cost Type")
    plt.tight_layout()
    st.pyplot(plt.gcf())

# ------------------------------
# Cost Metrics
# ------------------------------
if y is not None:
    st.subheader(f"{model_choice} Cost Metrics")
    cost_metrics = calculate_cost_metrics(y, df_processed["pred_label"])
    st.json(cost_metrics)

# === Cost heatmap (All Models) ===
if 'df_alerts' in locals() and df_alerts is not None:
    st.subheader("💰 Cost Heatmap (Top Alerts)")
    top_n_cost = st.slider("Top N transactions by cost", min_value=10, max_value=500, value=50)
    df_top = df_alerts.nlargest(top_n_cost, "total_cost")
    cost_cols = ["missed_fraud_cost", "false_alert_cost", "total_cost"]
    df_cost = df_top[cost_cols].T
    fig, ax = plt.subplots(figsize=(12,6))
    sns.heatmap(df_cost, annot=True, fmt=".0f", ax=ax)
    plt.title(f"Top {top_n_cost} transactions by total cost")
    st.pyplot(fig)

# === Top Regions by Fraud Risk ===
if 'df_processed' in locals() and 'location' in df_processed.columns:
    st.subheader("📍 Top Regions by Average Fraud Risk")
    fig_region = plot_top_by_region(df_processed)
    st.pyplot(fig_region)

# ==========================================
# SHAP Computation Function
# ==========================================
if model_choice == "XGBoost":
    st.info("Explaining XGBoost model predictions using SHAP values.")
elif model_choice == "Ensemble":
    st.info("Explaining Ensemble model predictions using weighted SHAP values from LGBM, XGBoost, and CatBoost.")

def compute_shap_for_model(model, X_sample, model_type='tree', ensemble_base=None, ensemble_meta=None):
    """Compute SHAP values for different model types."""
    try:
        if model_type == 'tree':
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

        elif model_type == 'ensemble':
            shap_values_list = []
            for name in ['lgbm', 'xgb', 'cat']:
                if name not in ensemble_base:
                    continue
                base_model = ensemble_base[name]
                expl = shap.TreeExplainer(base_model)
                shap_base = expl.shap_values(X_sample)

                coef = float(ensemble_meta.coef_[0][['lgbm', 'xgb', 'cat'].index(name)])
                shap_values_list.append(shap_base * coef)

            shap_values = np.sum(shap_values_list, axis=0)

        else:
            explainer = shap.Explainer(model.predict, X_sample)
            shap_values = explainer(X_sample)

        shap.summary_plot(shap_values, X_sample, show=False)
        plt.tight_layout()
        return shap_values

    except Exception as e:
        st.warning(f"❌ SHAP computation failed: {e}")
        return None

def shap_explain_single_transaction(model, X_sample, model_type='tree', index=0, ensemble_base=None, ensemble_meta=None):
    """Compute and display SHAP waterfall plot for a single transaction."""

    x_instance = X_sample.iloc[[index]]

    # Compute SHAP values based on model type
    if model_type == 'tree':
        expl = shap.TreeExplainer(model)
        shap_values = expl(x_instance)
    elif model_type == 'ensemble':
        shap_values_list = []
        for name in ['lgbm', 'xgb', 'cat']:
            base_model = ensemble_base[name]
            expl = shap.TreeExplainer(base_model)
            shap_base = expl(x_instance)
            coef = float(np.ravel(ensemble_meta.coef_)[['lgbm','xgb','cat'].index(name)])
            shap_values_list.append(shap_base.values * coef)
        shap_values = shap.Explanation(
            values=np.sum(shap_values_list, axis=0),
            base_values=expl.expected_value,
            data=x_instance.values,
            feature_names=x_instance.columns
        )
    else:
        expl = shap.Explainer(model.predict, X_sample)
        shap_values = expl(x_instance)

    # Display waterfall plot
    st.write("### 🔎 SHAP Waterfall Explanation for Selected Transaction")
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.waterfall_plot(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)
    plt.close()

if model_choice in ["XGBoost", "Ensemble"] and 'X' in locals():
    # Take a small sample for SHAP to keep runtime manageable
    X_sample = X.sample(min(300, len(X)), random_state=42)

    st.subheader("🔍 SHAP Feature Importance")

    # === 3️⃣ SHAP Single Transaction Explanation ===
    st.subheader("🧩 Individual Transaction Explanation")
    
    # User picks a transaction
    idx = st.number_input("Select transaction index for explanation", min_value=0, max_value=len(X_sample)-1, value=0, step=1)
    try:
        shap_explain_single_transaction(
            model=None,  # not used directly
            X_sample=X_sample,
            model_type="ensemble" if model_choice == "Ensemble" else "tree",
            index=int(idx),
            ensemble_base=ensemble_base if model_choice == "Ensemble" else None,
            ensemble_meta=ensemble_meta if model_choice == "Ensemble" else None
            )
    except Exception as e:
        st.warning(f"Unable to generate SHAP waterfall plot: {e}")

    try:
        if model_choice == "Ensemble":
            shap_values = compute_shap_for_model(
                X_sample = sample_df[numeric_features]
                X_sample=X_sample,
                model_type="ensemble",
                ensemble_base=ensemble_base,
                ensemble_meta=ensemble_meta
            )
        else:  # XGBoost
            shap_values = compute_shap_for_model(
                xgb_model, X_sample, model_type="tree"
            )

        if shap_values is not None:
            st.pyplot(plt.gcf(), clear_figure=True)

    except Exception as e:
        st.warning(f"SHAP visualization not available: {e}")

# === Single Transaction Prediction ===
st.subheader("🧮 Predict Single Transaction")
st.write("Enter feature values manually to get a real-time fraud prediction.")

categorical_prefixes = ["merchant_type", "payment_method", "location"]
categorical_features = []
for prefix in categorical_prefixes:
    categorical_features += [f for f in feature_columns if f.startswith(prefix + "_")]
numeric_features = [f for f in feature_columns if f not in categorical_features]

with st.form("single_txn_form"):
    cat_inputs = {}
    for prefix in categorical_prefixes:
        options = [f.split("_", 1)[1] for f in feature_columns if f.startswith(prefix + "_")]
        if options:
            cat_inputs[prefix] = st.selectbox(f"{prefix.replace('_',' ').title()}", options)
    num_inputs = {f: st.number_input(f"{f}", value=0.0) for f in numeric_features}
    submitted = st.form_submit_button("Predict Fraud Risk")

if submitted:
    sample = {**num_inputs}
    for prefix, value in cat_inputs.items():
        col_name = f"{prefix}_{value}"
        sample[col_name] = 1
        for f in feature_columns:
            if f.startswith(prefix + "_") and f != col_name:
                sample[f] = 0
    sample_df = pd.DataFrame([sample]).reindex(columns=feature_columns, fill_value=0)
    score = compute_single_score(sample_df, model_choice, numeric_features, w_xgb, w_iso)
    st.metric("Fraud Risk Score", f"{score:.3f}")
    st.markdown(f"<div style='color: {'green' if score < threshold else 'red'}; font-weight:bold;'>"
                f"{'🟢 Normal' if score < threshold else '🔴 Potential Fraud'}</div>", unsafe_allow_html=True)
