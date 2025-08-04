import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_models(X_train, y_train, model_type="Random Forest", model_params=None):
    """
    Train machine learning models for house price prediction
    
    Args:
        X_train: Training features
        y_train: Training targets
        model_type: Type of model to train ("Random Forest" or "Linear Regression")
        model_params: Dictionary of model parameters
    
    Returns:
        model: Trained model
        scaler: Fitted scaler (None for Random Forest)
    """
    if model_params is None:
        model_params = {}
    
    scaler = None
    
    if model_type == "Random Forest":
        # Random Forest doesn't require scaling
        model = RandomForestRegressor(**model_params)
        model.fit(X_train, y_train)
    
    elif model_type == "Linear Regression":
        # Linear Regression benefits from feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model = LinearRegression(**model_params)
        model.fit(X_train_scaled, y_train)
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model, scaler

def evaluate_model(model, X_test, y_test, scaler=None):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        scaler: Fitted scaler (if used)
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
    else:
        y_pred = model.predict(X_test)
    
    metrics = {
        'r2_score': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'predictions': y_pred
    }
    
    return metrics

def get_feature_importance(model, feature_names):
    """
    Get feature importance from Random Forest model
    
    Args:
        model: Trained Random Forest model
        feature_names: List of feature names
    
    Returns:
        pd.DataFrame: DataFrame with features and their importance scores
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not have feature_importances_ attribute")
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return importance_df

def predict_single(model, input_data, scaler=None):
    """
    Make prediction for a single instance
    
    Args:
        model: Trained model
        input_data: Input features (numpy array or list)
        scaler: Fitted scaler (if used)
    
    Returns:
        float: Predicted house value
    """
    input_array = np.array(input_data).reshape(1, -1)
    
    if scaler is not None:
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]
    else:
        prediction = model.predict(input_array)[0]
    
    return prediction

def get_prediction_confidence(model, input_data, scaler=None, n_estimators=None):
    """
    Get prediction confidence interval for Random Forest models
    
    Args:
        model: Trained Random Forest model
        input_data: Input features
        scaler: Fitted scaler (if used)
        n_estimators: Number of estimators (for Random Forest)
    
    Returns:
        dict: Dictionary with prediction, lower_bound, upper_bound
    """
    if not hasattr(model, 'estimators_'):
        # For non-ensemble models, return simple prediction
        prediction = predict_single(model, input_data, scaler)
        return {
            'prediction': prediction,
            'lower_bound': prediction * 0.9,  # Simple ±10% interval
            'upper_bound': prediction * 1.1
        }
    
    input_array = np.array(input_data).reshape(1, -1)
    
    if scaler is not None:
        input_scaled = scaler.transform(input_array)
        predictions = [tree.predict(input_scaled)[0] for tree in model.estimators_]
    else:
        predictions = [tree.predict(input_array)[0] for tree in model.estimators_]
    
    mean_pred = np.mean(predictions)
    std_pred = np.std(predictions)
    
    return {
        'prediction': mean_pred,
        'lower_bound': mean_pred - 1.96 * std_pred,  # 95% confidence interval
        'upper_bound': mean_pred + 1.96 * std_pred
    }
