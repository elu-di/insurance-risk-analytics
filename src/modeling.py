import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor, XGBClassifier

def prepare_data_for_modeling(data: pd.DataFrame, target_col: str):
    """Prepare data for modeling."""
    drop_cols = ['CustomerID', 'TransactionDate', 'TotalClaims', 'TotalPremium', 'Margin', 'Claim_Occurred']
    existing_drops = [c for c in drop_cols if c in data.columns]
    X = data.drop(columns=existing_drops)
    
    # Handle missing values
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna('Unknown')
    
    y = data[target_col]
    
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ])
    
    X_processed = preprocessor.fit_transform(X)
    X_processed = pd.DataFrame(X_processed, columns=preprocessor.get_feature_names_out())
    
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, preprocessor

def evaluate_regression(model, X_test, y_test):
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    return {'RMSE': rmse, 'R2': r2}

def evaluate_classification(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    return {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1}

def train_and_evaluate_models(X_train, X_test, y_train, y_test, task_type='regression'):
    if task_type == 'regression':
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=50, random_state=42)
        }
        eval_func = evaluate_regression
    else:
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
            'XGBoost': XGBClassifier(n_estimators=50, random_state=42)
        }
        eval_func = evaluate_classification
        
    results = {}
    best_model = None
    best_score = float('inf') if task_type == 'regression' else -1
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = eval_func(model, X_test, y_test)
        results[name] = metrics
        
        # Simple selection based on RMSE or F1
        if task_type == 'regression':
            if metrics['RMSE'] < best_score:
                best_score = metrics['RMSE']
                best_model = model
        else:
            if metrics['F1'] > best_score:
                best_score = metrics['F1']
                best_model = model
                
    return results, best_model
