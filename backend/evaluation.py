# backend/evaluation.py
from __future__ import annotations
import os
import joblib
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    precision_score, recall_score, f1_score, accuracy_score, precision_recall_curve
)
from sklearn.exceptions import UndefinedMetricWarning
import warnings

# -------------------------------
# Setup logging
# -------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# -------------------------------
# Utility: Safe AUC computation
# -------------------------------
def safe_auc(y_true, y_pred_proba):
    """Compute ROC-AUC safely; return np.nan if undefined."""
    try:
        return roc_auc_score(y_true, y_pred_proba)
    except Exception:
        warnings.warn("ROC-AUC failed due to constant predictions.", UndefinedMetricWarning)
        return np.nan
    
# -------------------------------
# Load models
# -------------------------------
def load_models():
    """Load all required models and artifacts."""
    xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgb_model.joblib"))
    iso_model = joblib.load(os.path.join(MODEL_DIR, "iso_model.joblib"))
    iso_scaler = joblib.load(os.path.join(MODEL_DIR, "iso_scaler.joblib"))
    iso_features = joblib.load(os.path.join(MODEL_DIR, "iso_training_columns.joblib"))
    numeric_features = joblib.load(os.path.join(MODEL_DIR, "iso_numeric_features.joblib"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_columns.joblib"))
    ensemble_base = joblib.load(os.path.join(MODEL_DIR, 'ensemble_base_models.joblib'))
    ensemble_meta = joblib.load(os.path.join(MODEL_DIR, 'ensemble_meta_model.joblib'))
    return xgb_model, iso_model, iso_features, iso_scaler, feature_cols, numeric_features, ensemble_base, ensemble_meta

# -------------------------------
# Prepare dataset
# -------------------------------
def prepare_dataset(df, target_col="Class", drop_cols=None):
    df = df.copy()
    if drop_cols is None:
        drop_cols = ["transaction_id", "timestamp", "Time"]
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])
    return X, y

# -------------------------------
# XGBoost Evaluation
# -------------------------------
def evaluate_xgb(xgb_model, X, y):
    logger.info("Evaluating XGBoost model...")
    y_pred_proba = xgb_model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    metrics = {
        "AUC": safe_auc(y, y_pred_proba),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "ConfusionMatrix": confusion_matrix(y, y_pred),
        "ClassificationReport": classification_report(y, y_pred, zero_division=0)
    }
    return y_pred_proba, metrics

# -------------------------------
# IsolationForest Evaluation
# -------------------------------
def evaluate_iso(iso_model, iso_scaler, X, y, numeric_features, iso_features=None):
    """Evaluate IsolationForest model using only numeric features."""
    logger.info("Evaluating IsolationForest model...")
    
    # Select only numeric features
    X_numeric = X[numeric_features]
    
    # Compute anomaly scores 
    iso_scores = -iso_model.decision_function(X_numeric).reshape(-1, 1)
    iso_scores_scaled = iso_scaler.transform(iso_scores).ravel()

    # Determine threshold based on contamination
    contamination = getattr(iso_model, "contamination", 0.05)
    threshold = np.percentile(iso_scores_scaled, 100 * (1 - contamination))
    y_pred = (iso_scores_scaled >= threshold).astype(int)

    metrics = {
        "AUC": safe_auc(y, iso_scores_scaled),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "ConfusionMatrix": confusion_matrix(y, y_pred)
    }
    
    return iso_scores_scaled, metrics

# -------------------------------
# Ensemble Evaluation
# -------------------------------
def evaluate_ensemble(ensemble_base, ensemble_meta, X, y):
    logger.info("Evaluating Ensemble model...")
    oof_preds = [ensemble_base[name].predict_proba(X)[:, 1] for name in ['lgbm', 'xgb', 'cat']]
    meta_X = np.vstack(oof_preds).T
    y_pred_proba = ensemble_meta.predict_proba(meta_X)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    metrics = {
        "AUC": safe_auc(y, y_pred_proba),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "ConfusionMatrix": confusion_matrix(y, y_pred),
        "ClassificationReport": classification_report(y, y_pred, zero_division=0)
    }
    return y_pred_proba, metrics

# -------------------------------
# Hybrid scoring
# -------------------------------
def hybrid_score(xgb_model, iso_model, X, w_xgb=0.7, w_iso=0.3,
                 iso_score_scaler=None, numeric_features=None, iso_features=None,
                 model_type='xgb', ensemble_base=None, ensemble_meta=None):
    """Compute hybrid scores for model_type='xgb' or 'ensemble'."""
    if model_type == 'xgb':
        p_fraud = xgb_model.predict_proba(X)[:, 1]
    elif model_type == 'ensemble':
        oof_preds = [ensemble_base[name].predict_proba(X)[:, 1] for name in ['lgbm', 'xgb', 'cat']]
        meta_X = np.vstack(oof_preds).T
        p_fraud = ensemble_meta.predict_proba(meta_X)[:, 1]
    else:
        raise ValueError("Invalid model_type")

    p_fraud = np.clip(p_fraud, 1e-6, 1 - 1e-6)

    # IsolationForest scoring   
    X_iso = X.reindex(columns=iso_features, fill_value=0)
    X_numeric = X_iso[numeric_features] if numeric_features else X_iso
    iso_raw = -iso_model.decision_function(X_numeric).reshape(-1, 1)
    iso_scaled = iso_score_scaler.transform(iso_raw).ravel()
    
    # Temperature calibration
    T = 1.2
    p_fraud_calibrated = 1 / (1 + np.exp(-np.log(p_fraud / (1 - p_fraud)) / T))

    hybrid = np.clip(w_xgb * p_fraud_calibrated + w_iso * iso_scaled, 0, 1)
    return hybrid, p_fraud_calibrated, iso_scaled

# -------------------------------
# Precision-oriented hybrid tuning
# -------------------------------
def tune_hybrid_weights(y_true, model_type='xgb', X=None, xgb_model=None,
                        ensemble_base=None, ensemble_meta=None,
                        iso_model=None, iso_score_scaler=None,
                        numeric_features=None, iso_features=None,
                        avg_fraud_amount=2000, fp_cost=20):
    """
    Tune hybrid weights for XGBoost or Ensemble + IsolationForest hybrid.
    Returns optimal (w_xgb, w_iso) pair minimizing total fraud cost.

    Parameters
    ----------
    y_true : array-like
        Ground truth fraud labels (0 = legit, 1 = fraud)
    model_type : str
        'xgb' or 'ensemble'
    xgb_model : optional
        Trained XGBoost model
    ensemble_base : dict, optional
        Base models used in the ensemble
    ensemble_meta : model, optional
        Meta model for ensemble
    X : pd.DataFrame, optional
        Input features for computing predictions
    iso_model, iso_score_scaler : optional
        IsolationForest model and its MinMaxScaler
    numeric_features, iso_features : list
        Columns used during IsolationForest training
    avg_fraud_amount : float
        Estimated average fraud transaction value
    fp_cost : float
        Cost of false positives (manual review, alerts)
    """
    #Grid search for optimal hybrid weights minimizing cost.
    if X is None:
        raise ValueError("X is required to compute hybrid weights")

    logger.info(f"Tuning hybrid weights for {model_type.upper()} model...")
    
    # ----- Step 1: Compute p_fraud -----
    if model_type == 'xgb':
        p_fraud = xgb_model.predict_proba(X)[:, 1]
    elif model_type == 'ensemble':
        oof_preds = [ensemble_base[name].predict_proba(X)[:, 1] for name in ['lgbm', 'xgb', 'cat']]
        meta_X = np.vstack(oof_preds).T
        p_fraud = ensemble_meta.predict_proba(meta_X)[:, 1]
    else:
        raise ValueError("Invalid model_type")

    # ----- Step 2: Compute iso_scaled -----
    X_iso = X.reindex(columns=iso_features, fill_value=0)
    X_numeric = X_iso[numeric_features] if numeric_features else X_iso
    iso_raw = -iso_model.decision_function(X_numeric).reshape(-1, 1)
    iso_scaled = iso_score_scaler.transform(iso_raw).ravel()
    
    # ----- Step 3: Grid search -----
    grid = [(w_x, 1 - w_x) for w_x in np.arange(0.5, 1.0, 0.05)]
    best_cost = float('inf')
    best_weights = (0.7, 0.3)

    for w_xgb, w_iso in grid:
        hybrid_scores = np.clip(w_xgb * p_fraud + w_iso * iso_scaled, 0, 1)
        prec, rec, thresh = precision_recall_curve(y_true, hybrid_scores)
        if len(thresh) == 0:
            threshold = 0.5
        else:
            f1_arr = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-8)
            threshold = thresh[np.argmax(f1_arr)]

        y_pred = (hybrid_scores >= threshold).astype(int)
        FN = np.sum((y_true == 1) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        precision_val = precision_score(y_true, y_pred, zero_division=0)
        penalty = (1 - precision_val) * 10
        total_cost = FN * avg_fraud_amount + FP * (fp_cost * (5 + penalty))

        logger.info(f"GRID w_xgb={w_xgb:.2f}, w_iso={w_iso:.2f} → Precision={precision_val:.3f}, Cost={total_cost:.2f}")
        if total_cost < best_cost:
            best_cost = total_cost
            best_weights = (w_xgb, w_iso)

    logger.info(f"Optimal weights → w_xgb={best_weights[0]:.2f}, w_iso={best_weights[1]:.2f}, cost={best_cost:.2f}")
    return best_weights

# -------------------------------
# Model evaluation utilities
# -------------------------------
def evaluate_model(y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    cls_report = classification_report(
        y_true, y_pred, target_names=["Legit (0)", "Fraud (1)"], output_dict=True, zero_division=0
    )
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": safe_auc(y_true, y_pred_proba),
        "ConfusionMatrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "ClassificationReport": cls_report
    }

# -------------------------------
# Cost-sensitive metrics
# -------------------------------
def calculate_cost_metrics(y_true, y_pred, amounts=None, fp_cost=20):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    cost_missed_fraud = np.sum(amounts[(y_true==1)&(y_pred==0)]) if amounts is not None else fn*2000
    cost_false_alerts = fp * fp_cost
    total_cost = cost_missed_fraud + cost_false_alerts
    return {
        "Missed Fraud Count": fn,
        "False Alerts Count": fp,
        "Cost of Missed Fraud": cost_missed_fraud,
        "Cost of False Alerts": cost_false_alerts,
        "Total Estimated Cost Impact": total_cost
    }

# -------------------------------
# Risk labeling (0–1 scale)
# -------------------------------
def risk_label(score: float, scale: str = "0-1") -> str:
    pct = score * 100 if scale == "0-1" else score
    if pct < 30:
        return "Green"
    elif pct < 70:
        return "Yellow"
    return "Red"

# -------------------------------
# Full evaluation report
# -------------------------------
def full_evaluation_report(y_true, y_pred_proba, amounts=None, threshold=0.5,
                           avg_fraud_amount=2000, fp_cost=20):
    y_pred = (y_pred_proba >= threshold).astype(int)
    model_metrics = evaluate_model(y_true, y_pred_proba, threshold)
    if amounts is not None:
        amounts = np.array(amounts)
    else:
        amounts = np.ones_like(y_true) * avg_fraud_amount
    cost_metrics = calculate_cost_metrics(y_true, y_pred, amounts=amounts, fp_cost=fp_cost)
    return {**model_metrics, **cost_metrics}

def tune_threshold_by_cost(y_true, scores, amounts=None, fp_cost=20, avg_fraud_amount=2000):
    if amounts is None:
        amounts = np.ones_like(y_true) * avg_fraud_amount
    thresholds = np.linspace(0,1,101)
    best = {'threshold':0.5, 'cost':np.inf}
    for t in thresholds:
        y_pred = (scores >= t).astype(int)
        FN = np.sum((y_true==1)&(y_pred==0))
        FP = np.sum((y_true==0)&(y_pred==1))
        cost = np.sum(amounts[(y_true==1)&(y_pred==0)]) + FP*fp_cost
        if cost < best['cost']:
            best = {'threshold':t, 'cost':cost}
    return best

# -------------------------------
# Main execution (for CLI/local testing)
# -------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("Running backend evaluation pipeline...")

    # Load models
    try:
        xgb_model, iso_model, iso_features, iso_scaler, feature_cols, numeric_features, ensemble_base, ensemble_meta = load_models()
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        sys.exit(1)

    #Load test data
    test_data = pd.read_parquet(os.path.join(DATA_DIR, "test_data.parquet"))
    X_test, y_test = prepare_dataset(test_data)
    logger.info(f"Loaded test data → X_test: {X_test.shape}, y_test: {y_test.shape}")

    # ----- XGBoost Evaluation -----
    y_pred_xgb, metrics_xgb = evaluate_xgb(xgb_model, X_test, y_test)
    logger.info(f"[XGB] AUC={metrics_xgb['AUC']:.4f}, F1={metrics_xgb['F1']:.4f}, "
                f"Precision={metrics_xgb['Precision']:.3f}, Recall={metrics_xgb['Recall']:.3f}")
    
    # Prepare X_test for IsolationForest
    X_test_iso = X_test.reindex(columns=iso_features, fill_value=0)
    # ----- IsolationForest Evaluation -----
    iso_scores, metrics_iso = evaluate_iso(
        iso_model, 
        iso_scaler,
        X_test,
        y_test,
        numeric_features=numeric_features
    )

    logger.info(f"[ISO] AUC={metrics_iso['AUC']:.4f}, F1={metrics_iso['F1']:.4f}, "
            f"Precision={metrics_iso['Precision']:.3f}, Recall={metrics_iso['Recall']:.3f}")

    # ----- Ensemble Evaluation -----
    y_pred_ens, metrics_ens = evaluate_ensemble(ensemble_base, ensemble_meta, X_test, y_test)
    logger.info(f"[ENSEMBLE] AUC={metrics_ens['AUC']:.4f}, F1={metrics_ens['F1']:.4f}, "
                f"Precision={metrics_ens['Precision']:.3f}, Recall={metrics_ens['Recall']:.3f}")

    # ----- Hybrid Risk Score -----
    X_test_iso = X_test.reindex(columns=iso_features, fill_value=0)
    hybrid_scores, _, _ = hybrid_score(
    xgb_model, iso_model, X_test,
    w_xgb=0.7, w_iso=0.3,
    iso_score_scaler=iso_scaler,
    numeric_features=numeric_features,
    iso_features=iso_features,
    model_type='ensemble',
    ensemble_base=ensemble_base,
    ensemble_meta=ensemble_meta
    )
    logger.info(f"Hybrid risk scores computed for {len(hybrid_scores)} records.")

    # ----- Weight Tuning -----
    best_wxgb, best_wiso = tune_hybrid_weights(
        y_true=y_test,
        model_type='ensemble',
        X=X_test,
        xgb_model=xgb_model,
        ensemble_base=ensemble_base,
        ensemble_meta=ensemble_meta,
        iso_model=iso_model,
        iso_score_scaler=iso_scaler,
        numeric_features=numeric_features,
        iso_features=iso_features
    )
    logger.info(f"Optimal Hybrid Weights → w_xgb={best_wxgb:.2f}, w_iso={best_wiso:.2f}")
    
    
    # ----- Cost-optimized threshold -----
    best_thresh = tune_threshold_by_cost(
        y_true=y_test,
        scores=hybrid_scores,
        fp_cost=20,
        avg_fraud_amount=2000
    )
    logger.info(f"Optimal threshold for hybrid scores: {best_thresh['threshold']:.3f}, Estimated Cost: {best_thresh['cost']:.2f}")

    # ----- Apply threshold to get binary predictions -----
    y_pred_hybrid = (hybrid_scores >= best_thresh['threshold']).astype(int)

    # ----- Map hybrid scores to risk labels -----
    risk_labels = [risk_label(s) for s in hybrid_scores]

    # ----- Full Evaluation -----
    full_report = full_evaluation_report(y_test, hybrid_scores)
    logger.info("Full Evaluation Summary:")
    for k, v in full_report.items():
        if isinstance(v, (float, int)):
            logger.info(f"  {k:<30}: {v:.4f}")
        elif isinstance(v, dict):
            logger.info(f"  {k}: {v}")

    # Example: Show risk labels for first 10 records -----
    logger.info(f"Sample risk labels for first 10 transactions: {risk_labels[:10]}")