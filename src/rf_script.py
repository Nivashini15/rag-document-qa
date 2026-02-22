from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def train_stock_model(master_df):
    # 1. Prepare Features and Target
    # We use sentiment and historical price to predict the NEXT day's return
    X = master_df[['Scaled_Close', 'Scaled_Vol', 'Sentiment_Score', 'Prev_Day_Sentiment']]
    y = master_df['Returns'].shift(-1).fillna(0) # Predict tomorrow's return

    # 2. Time-Series Split (Crucial for Finance!)
    # We don't use random shuffle because the order of days matters.
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 3. Initialize Random Forest with optimized Hyperparameters
    rf_model = RandomForestRegressor(
        n_estimators=200,      # Number of trees
        max_depth=10,          # Limits depth to prevent overfitting
        min_samples_leaf=5,    # Minimum samples per leaf for stability
        random_state=42
    )

    # 4. Training
    rf_model.fit(X_train, y_train)

    # 5. Evaluation
    predictions = rf_model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Model Training Complete.\nMAE: {mae:.4f}\nR-Squared: {r2:.4f}")
    
    return rf_model, predictions

# --- Example Execution ---
# model, preds = train_stock_model(fused_dataframe)