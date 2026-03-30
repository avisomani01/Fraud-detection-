# backend/model_training.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")

# Paths (use os.path.join for portability)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # backend folder
PROJECT_ROOT = os.path.dirname(BASE_DIR)              # parent of backend (project root)
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MASTER_PATH = os.path.join(PROJECT_ROOT, "data", "master_dataset.parquet")

def load_master(path=MASTER_PATH):
    df = pd.read_parquet(path)
    print(f"[INFO] Master dataset loaded: {df.shape}")
    return df

def prepare_dataset(df, target_col="Class", drop_cols=None):
    """Prepare X, y. Drop identifiers and timestamp-like columns."""
    df = df.copy()
    if drop_cols is None:
        drop_cols = ["transaction_id", "timestamp", "Time"]  # adjust if necessary
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])
    print(f"[INFO] Prepared X ({X.shape}) and y ({y.shape})")
    return X, y

def train_test_split_stratified(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

def train_xgb(X_train, y_train, X_val=None, y_val=None, model_path=os.path.join(MODEL_DIR, "xgb_model.joblib")):
    """Train cost-sensitive XGBoost with CV and save model."""
    pos = np.sum(y_train == 1)
    neg = np.sum(y_train == 0)
    imbalance_ratio = neg / (pos + 1e-6)

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        use_label_encoder=False,
        scale_pos_weight=imbalance_ratio * 1.5,  # bias for precision (tweakable)
        n_estimators=400,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        reg_alpha=1.0,
        random_state=42,
        n_jobs=-1
    )

    # 5-fold CV
    print("[INFO] Performing 5-Fold CV...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_metrics = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model.fit(X_tr, y_tr)
        y_val_pred = model.predict_proba(X_val_fold)[:, 1]

        auc = roc_auc_score(y_val_fold, y_val_pred)
        prec = precision_score(y_val_fold, y_val_pred >= 0.5, zero_division=0)
        rec = recall_score(y_val_fold, y_val_pred >= 0.5, zero_division=0)
        f1 = f1_score(y_val_fold, y_val_pred >= 0.5, zero_division=0)
        cv_metrics.append((auc, prec, rec, f1))
        print(f"[FOLD {fold}] AUC={auc:.4f}, Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")

    avg_metrics = np.mean(cv_metrics, axis=0)
    print(f"\n[CV AVERAGE] AUC={avg_metrics[0]:.4f}, Precision={avg_metrics[1]:.3f}, Recall={avg_metrics[2]:.3f}, F1={avg_metrics[3]:.3f}")

    # Retrain final model on full train data
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, model_path)
    print(f"[INFO] XGBoost model trained and saved to {model_path}")
    return model

def train_isolation_forest(X_train, contamination=0.01, model_path=os.path.join(MODEL_DIR, "iso_model.joblib")):
    """Train IsolationForest on numeric features only."""
    # Select only numeric columns
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    X_train_numeric = X_train[numeric_features]
    
    iso = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    
    # Fit on numeric features only
    iso.fit(X_train_numeric)
    
    # Save model and feature lists
    joblib.dump(iso, model_path)
    joblib.dump(numeric_features, os.path.join(MODEL_DIR, "iso_numeric_features.joblib"))
    print(f"[INFO] IsolationForest trained and saved to {model_path}")
    print(f"[INFO] Numeric features saved to iso_numeric_features.joblib")

    # Compute scaled IsolationForest scores for hybrid evaluation
    iso_inv_train = -iso.decision_function(X_train_numeric).reshape(-1, 1)
    iso_scaler = MinMaxScaler().fit(iso_inv_train)
    joblib.dump(iso_scaler, os.path.join(MODEL_DIR, "iso_scaler.joblib"))
    print(f"[INFO] IsolationForest scaler saved → iso_scaler.joblib")
    return iso

def train_ensemble(X_train, y_train, model_dir=MODEL_DIR):
    # base learners
    lgbm = lgb.LGBMClassifier(n_estimators=300, random_state=42)
    xgb = XGBClassifier(n_estimators=300, random_state=42, use_label_encoder=False, eval_metric='auc')
    cat = CatBoostClassifier(verbose=0, iterations=300, random_state=42)
    
    # Fit base learners
    lgbm.fit(X_train, y_train)
    xgb.fit(X_train, y_train)
    cat.fit(X_train, y_train)
    
    joblib.dump({'lgbm': lgbm, 'xgb': xgb, 'cat': cat}, os.path.join(model_dir, 'ensemble_base_models.joblib'))
    print('[INFO] Saved ensemble base models')
    
    # Create meta-features by OOF predictions (stacking)
    oof_l = cross_val_predict(lgbm, X_train, y_train, cv=5, method='predict_proba')[:,1]
    oof_x = cross_val_predict(xgb, X_train, y_train, cv=5, method='predict_proba')[:,1]
    oof_c = cross_val_predict(cat, X_train, y_train, cv=5, method='predict_proba')[:,1]
    
    meta_X = np.vstack([oof_l, oof_x, oof_c]).T
    meta_model = LogisticRegression(max_iter=1000)
    meta_model.fit(meta_X, y_train)
    joblib.dump(meta_model, os.path.join(model_dir, 'ensemble_meta_model.joblib'))
    print('[INFO] Saved ensemble meta-model')
    return {'lgbm': lgbm, 'xgb': xgb, 'cat': cat}, meta_model


if __name__ == "__main__":
    # --------------------
    # Load & prepare data
    # --------------------
    df = load_master(MASTER_PATH)
    X, y = prepare_dataset(df, target_col="Class", drop_cols=["transaction_id", "timestamp", "Time"])
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, test_size=0.2)
    print(f"[INFO] Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    feature_cols = X_train.columns.tolist()

    # Save test set for later evaluation
    test_df = X_test.copy()
    test_df["Class"] = y_test.values
    test_path = os.path.join("data", "test_data.parquet")
    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    test_df.to_parquet(test_path)
    print(f"[INFO] Saved test data → {test_path}")

    # Additionally, save X_test and y_test separately for easier loading in evaluation
    X_test.to_parquet(os.path.join(MODEL_DIR, "X_test.parquet"), index=False)
    y_test.to_frame().to_parquet(os.path.join(MODEL_DIR, "y_test.parquet"), index=False)
    print("[INFO] Saved X_test and y_test separately in models folder")

    # Keep a copy of original train for IsolationForest
    X_train_orig = X_train.copy()
    y_train_orig = y_train.copy()
    X_train.to_parquet(os.path.join(MODEL_DIR, "X_train.parquet"), index=False)
    y_train.to_frame().to_parquet(os.path.join(MODEL_DIR, "y_train.parquet"), index=False)   
    X_train_orig.to_parquet(os.path.join(MODEL_DIR, "X_train_orig.parquet"), index=False)
    y_train_orig.to_frame().to_parquet(os.path.join(MODEL_DIR, "y_train_orig.parquet"), index=False)  

    # Save IsolationForest training columns
    iso_features = X_train_orig.columns.tolist()
    joblib.dump(iso_features, os.path.join(MODEL_DIR, "iso_training_columns.joblib"))
    print(f"[INFO] Saved IsolationForest training columns → {len(iso_features)} columns")

    # Save numeric columns for IsolationForest
    iso_numeric_features = X_train_orig.select_dtypes(include=[np.number]).columns.tolist()
    joblib.dump(iso_numeric_features, os.path.join(MODEL_DIR, "iso_numeric_features.joblib"))
    print(f"[INFO] Saved IsolationForest numeric features → {len(iso_numeric_features)} columns")

    # --------------------
    # Train models
    # --------------------
    xgb_model = train_xgb(X_train, y_train, X_val=X_test, y_val=y_test)
    iso_model = train_isolation_forest(X_train_orig, contamination=max(0.001, min(0.05, y_train_orig.mean() * 2)))

    # Save feature list for evaluation
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_columns.joblib"))
    print(f"[INFO] Saved feature list → {len(feature_cols)} columns.")

    ensemble_models, meta_model = train_ensemble(X_train, y_train)
    print("[INFO] Ensemble (stacked) model training complete.")


