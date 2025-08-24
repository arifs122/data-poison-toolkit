import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_absolute_error, r2_score
from scipy.stats import wasserstein_distance
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler


def evaluate_stealth(clean_series: pd.Series, poison_series: pd.Series) -> dict:
    """Measures the stealthiness of the attack with a single score by comparing statistical distributions."""
    if clean_series.dtype not in ['float64', 'int64']:
        return {"stealth_score": "N/A"}
        
    mean_shift_std = np.abs(poison_series.mean() - clean_series.mean()) / clean_series.std()
    std_shift_std = np.abs(poison_series.std() - clean_series.std()) / clean_series.std()
    wasserstein_dist = wasserstein_distance(clean_series.dropna(), poison_series.dropna())
    
    # Aggregate all metrics to create a single stealthiness score
    # The lower this value, the stealthier the attack.
    composite_stealth = mean_shift_std + std_shift_std + wasserstein_dist
    
    return {"composite_stealth": composite_stealth}

def evaluate_impact(clean_df: pd.DataFrame, poison_df: pd.DataFrame, target_col: str, task_type: str) -> dict:
    """
    Saldırının, temiz veriyle eğitilmiş modele kıyasla zehirli veriyle eğitilen modelin performansını nasıl etkilediğini ölçer.
    """
    # 1. Separate the data from the Clean DataFrame.
    X_clean = clean_df.drop(columns=[target_col])
    y_clean = clean_df[target_col]
    
    # 2. Separate features and labels from the poisoned DataFrame.
    X_poisoned = poison_df.drop(columns=[target_col])
    y_poisoned = poison_df[target_col]
    
    # 3. Split the data into training and test sets. Poisoned and clean sets should have the same indices.
    X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(
        X_clean, y_clean, test_size=0.3, random_state=0)
    
    # The poisoned training set should be split in the same way as the clean test set.
    X_train_poisoned = X_poisoned.loc[X_train_clean.index]
    y_train_poisoned = y_poisoned.loc[y_train_clean.index]

    # 4. Scale the data (fit on the training data and transform the test data).
    scaler = StandardScaler()
    X_train_clean_scaled = scaler.fit_transform(X_train_clean)
    X_test_clean_scaled = scaler.transform(X_test_clean)
    
    X_train_poisoned_scaled = scaler.fit_transform(X_train_poisoned)
    
    # 5. Select the model.
    if task_type == 'classification':
        model_clean = RandomForestClassifier(n_estimators=100, random_state=0)
        model_poisoned = RandomForestClassifier(n_estimators=100, random_state=0)
    elif task_type == 'regression':
        model_clean = RandomForestRegressor(n_estimators=100, random_state=0)
        model_poisoned = RandomForestRegressor(n_estimators=100, random_state=0)
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    # 6. Train models separately on clean and poisoned data.
    model_clean.fit(X_train_clean_scaled, y_train_clean)
    model_poisoned.fit(X_train_poisoned_scaled, y_train_poisoned)

    # 7. Evaluate the performance of both models on the clean test set.
    y_pred_clean_model = model_clean.predict(X_test_clean_scaled)
    y_pred_poisoned_model = model_poisoned.predict(X_test_clean_scaled)
    
    results = {}
    if task_type == 'classification':
        clean_score = accuracy_score(y_test_clean, y_pred_clean_model)
        poisoned_score = accuracy_score(y_test_clean, y_pred_poisoned_model)
        results["impact_score"] = clean_score - poisoned_score
        results["clean_accuracy"] = clean_score
        results["poisoned_accuracy"] = poisoned_score
    elif task_type == 'regression':
        clean_score = mean_absolute_error(y_test_clean, y_pred_clean_model)
        poisoned_score = mean_absolute_error(y_test_clean, y_pred_poisoned_model)
        results["impact_score"] = poisoned_score - clean_score
        results["clean_mae"] = clean_score
        results["poisoned_mae"] = poisoned_score

    return results

def evaluate_defense_performance(all_indices: pd.Index, poison_indices: pd.Index, detected_indices: pd.Index) -> dict:
    if isinstance(poison_indices, np.ndarray):
        poison_indices = pd.Index(poison_indices)
    elif isinstance(poison_indices, list):
        poison_indices = pd.Index(poison_indices)
        
    y_true = pd.Series(0, index=all_indices)
    if not poison_indices.empty:
        y_true.loc[poison_indices] = 1
    y_pred = pd.Series(0, index=all_indices)
    if not detected_indices.empty:
        y_pred.loc[detected_indices] = 1
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    return {"precision": precision, "recall": recall, "f1_score": f1}