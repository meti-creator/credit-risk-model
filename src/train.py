import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

def evaluate_model(y_true, y_pred, y_prob):
    """Calculates all 5 required assessment metrics."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1_Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_prob)
    }

def run_model_training_and_tracking():
    print("🏋️ Starting Model Training and MLflow Experiment Tracking...")
    
    # 1. Set up file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(script_dir), 'data', 'processed', 'model_ready_data.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Cannot find processed data at: {data_path}. Run data_processing.py first!")
        
    # 2. Load and Split Data
    df = pd.read_csv(data_path)
    X = df.drop(columns=['is_high_risk'])
    y = df['is_high_risk']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"📋 Data split complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 3. Initialize MLflow Experiment
    mlflow.set_experiment("Credit_Risk_Proxy_Modeling")
    
    best_f1 = -1
    best_run_id = None
    best_model_name = ""

    # =====================================================================
    # EXPERIMENT 1: LOGISTIC REGRESSION WITH GRID SEARCH TUNING
    # =====================================================================
    with mlflow.start_run(run_name="Logistic_Regression_GridSearch") as run:
        print("\n🔵 Training Model 1: Logistic Regression...")
        
        lr_base = LogisticRegression(max_iter=1000, random_state=42)
        lr_param_grid = {'C': [0.01, 0.1, 1.0, 10.0]}
        
        # Grid Search tuning
        grid_search = GridSearchCV(lr_base, lr_param_grid, cv=3, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_lr = grid_search.best_estimator_
        
        # Predict values and probabilities
        preds = best_lr.predict(X_test)
        probs = best_lr.predict_proba(X_test)[:, 1]
        
        # Compute metrics
        metrics = evaluate_model(y_test, preds, probs)
        
        # Log to MLflow
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_lr, artifact_path="model")
        print(f"✨ LR Complete. F1-Score: {metrics['F1_Score']:.4f}")
        
        if metrics['F1_Score'] > best_f1:
            best_f1 = metrics['F1_Score']
            best_run_id = run.info.run_id
            best_model_name = "LogisticRegression_Champion"

    # =====================================================================
    # EXPERIMENT 2: RANDOM FOREST WITH RANDOM SEARCH TUNING
    # =====================================================================
    with mlflow.start_run(run_name="Random_Forest_RandomSearch") as run:
        print("\n🟢 Training Model 2: Random Forest...")
        
        rf_base = RandomForestClassifier(random_state=42)
        rf_param_dist = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5]
        }
        
        # Random Search tuning
        random_search = RandomizedSearchCV(rf_base, rf_param_dist, n_iter=3, cv=3, scoring='f1', random_state=42, n_jobs=-1)
        random_search.fit(X_train, y_train)
        
        best_rf = random_search.best_estimator_
        
        # Predict values and probabilities
        preds = best_rf.predict(X_test)
        probs = best_rf.predict_proba(X_test)[:, 1]
        
        # Compute metrics
        metrics = evaluate_model(y_test, preds, probs)
        
        # Log to MLflow
        mlflow.log_params(random_search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_rf, artifact_path="model")
        print(f"✨ RF Complete. F1-Score: {metrics['F1_Score']:.4f}")
        
        if metrics['F1_Score'] > best_f1:
            best_f1 = metrics['F1_Score']
            best_run_id = run.info.run_id
            best_model_name = "RandomForest_Champion"

    # =====================================================================
    # AUTOMATED MODEL REGISTRY FOR THE WINNER
    # =====================================================================
    print(f"\n🏆 Registering the Champion Model to the MLflow Registry...")
    model_uri = f"runs:/{best_run_id}/model"
    registered_model = mlflow.register_model(model_uri, best_model_name)
    print(f"🎉 Successfully registered '{best_model_name}' version 1 in the VIP Lounge!")

if __name__ == "__main__":
    run_model_training_and_tracking()
    