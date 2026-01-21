"""
Stock Price Prediction – Futures First Assignment

FINAL DECISIONS FROM EDA
------------------------

TARGET:
- Next day stock price

FEATURES INCLUDED (based on EDA + Mutual Information):
1. External Data Dynamics
   - data_change_lag1
   - 3-day rolling mean of data change
   - 7-day rolling mean of data change
   - 3-day rolling standard deviation of data change

2. Minimal price memory (baseline, not leakage)
   - previous day stock price

NOTES:
- Individual features show weak linear correlation.
- Mutual Information reveals non-linear, distributed signal.
- Rolling trends and volatility carry more information than raw lags.
- Final feature selection prioritizes generalization over complexity.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor



def load_data():
    """
    Loads input datasets, parses dates, and merges them on Date.
    Returns a single dataframe sorted by time.
    """

    # Read input files
    data_df = pd.read_csv("input/Data 2.csv")
    stock_df = pd.read_csv("input/StockPrice.csv")

    # Convert Date columns
    data_df["Date"] = pd.to_datetime(data_df["Date"])
    stock_df["Date"] = pd.to_datetime(stock_df["Date"])

    # Sort by date
    data_df = data_df.sort_values("Date").reset_index(drop=True)
    stock_df = stock_df.sort_values("Date").reset_index(drop=True)

    # Merge on Date (inner join)
    df = pd.merge(stock_df, data_df, on="Date", how="inner")

    # Final sort
    df = df.sort_values("Date").reset_index(drop=True)

    return df
def build_features(df):
    """
    Builds features justified through EDA and Mutual Information analysis.
    Prepares the final modeling dataset without introducing technical
    pattern assumptions (e.g., double bottom).
    """

    df = df.copy()

    # --------------------------------------------------
    # 1. Day-over-day changes (core signal)
    # --------------------------------------------------
    df["data_change"] = df["Data"].pct_change()
    df["price_change"] = df["Price"].diff()

    # --------------------------------------------------
    # 2. Lagged Data change (short-term memory)
    # --------------------------------------------------
    df["data_change_lag1"] = df["data_change"].shift(1)

    # --------------------------------------------------
    # 3. Smoothed Data signals (trend capture)
    # --------------------------------------------------
    df["data_change_ma3"] = df["data_change"].rolling(window=3).mean()
    df["data_change_ma7"] = df["data_change"].rolling(window=7).mean()

    # --------------------------------------------------
    # 4. Data volatility (market uncertainty)
    # --------------------------------------------------
    df["data_change_std3"] = df["data_change"].rolling(window=3).std()

    # --------------------------------------------------
    # 5. Minimal price memory (baseline persistence)
    # --------------------------------------------------
    df["price_lag1"] = df["Price"].shift(1)

    # --------------------------------------------------
    # 6. Target variable
    # --------------------------------------------------
    df["target"] = df["Price"].shift(-1)

    # --------------------------------------------------
    # 7. Clean dataset
    # --------------------------------------------------
    df = df.dropna().reset_index(drop=True)

    return df



"""
def build_features(df):
   

    df = df.copy()

    # -----------------------------
    # External Data Features
    # -----------------------------
    df["data_change"] = df["Data"].pct_change()

    df["data_change_lag1"] = df["data_change"].shift(1)
    df["data_change_ma3"] = df["data_change"].rolling(3).mean()
    df["data_change_ma7"] = df["data_change"].rolling(7).mean()
    df["data_change_std3"] = df["data_change"].rolling(3).std()

    # -----------------------------
    # Minimal Price Memory
    # -----------------------------
    df["price_lag1"] = df["Price"].shift(1)

    # Temporary price change (not kept as standalone feature)
    price_change = df["Price"].diff()

    # -----------------------------
    # Support / Reversal Structure
    # -----------------------------
    df["local_bottom"] = df["Price"].rolling(window=20).min()
    df["dist_from_bottom"] = df["Price"] - df["local_bottom"]

    df["is_pullback"] = (
        (df["dist_from_bottom"] > 0) &
        (price_change.shift(1) > 0)
    ).astype(int)

    # -----------------------------
    # Target
    # -----------------------------
    df["target"] = df["Price"].shift(-1)

    # -----------------------------
    # Final Cleanup
    # -----------------------------
    df = df.dropna().reset_index(drop=True)

    return df

    """



def time_train_test_split(df, test_size=0.15):
    """
    Splits data into train and test sets based on time order.
    """

    split_index = int(len(df) * (1 - test_size))

    X = df.drop(columns=["target", "Date"])

    y = df["target"]

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test

def train_linear_regression(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae = mean_absolute_error(y_test, y_test_pred)

    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    return model, rmse, mae, r2_train, r2_test

def train_scaled_linear_regression(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae = mean_absolute_error(y_test, y_test_pred)

    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    return model, scaler, rmse, mae, r2_train, r2_test


def train_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae = mean_absolute_error(y_test, y_test_pred)

    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    return model, rmse, mae, r2_train, r2_test

def train_gradient_boosting(X_train, y_train, X_test, y_test):
    """
    Trains a Gradient Boosting Regressor and evaluates it.
    """

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae = mean_absolute_error(y_test, y_test_pred)

    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    return model, rmse, mae, r2_train, r2_test



if __name__ == "__main__":

    # -----------------------------
    # 1. Load & prepare data
    # -----------------------------
    df = load_data()
    df = build_features(df)
    import os
    RESULTS_DIR = "results"
    os.makedirs(RESULTS_DIR, exist_ok=True)


    # -----------------------------
    # SIGNAL-ONLY SUBMISSION MODEL
    # Uses only Data-derived features (no price_lag1 or Price)
    # -----------------------------
    # Create ΔPrice target (next day change). We keep original 'target' for baseline untouched.
    df["target_delta"] = df["Price"].shift(-1) - df["Price"]

    # Signal features (only Data-derived)
    signal_features = [
        "data_change_lag1",
        "data_change_ma3",
        "data_change_ma7",
        "data_change_std3"
    ]

    # Build a temporary df suitable for time_train_test_split:
    # it expects a 'Date' column and a 'target' column named 'target',
    # and it will drop 'target' and 'Date' to form X internally.
    df_signal = df[["Date"] + signal_features + ["target_delta"]].copy()
    df_signal = df_signal.rename(columns={"target_delta": "target"})
    df_signal = df_signal.dropna().reset_index(drop=True)


    # Time split for signal model (uses your existing function)
    X_train_s, X_test_s, y_train_s, y_test_s = time_train_test_split(df_signal)  # default test_size used

    print("\nSignal Model split shapes:",
        "X_train_s:", X_train_s.shape, "X_test_s:", X_test_s.shape,
        "y_train_s:", y_train_s.shape, "y_test_s:", y_test_s.shape)

    # Train signal model (reuse your function)
    signal_model, rmse_sig, mae_sig, r2_sig_train, r2_sig_test = train_linear_regression( X_train_s, y_train_s, X_test_s, y_test_s)

    # Get predictions (ΔPrice) and reconstruct price_t+1 from today's price
    y_pred_delta = signal_model.predict(X_test_s)

    # base_price for reconstruction = Price_t (aligned to rows in X_test_s)
    base_price = df.loc[X_test_s.index, "Price"].values
    predicted_price_from_signal = base_price + y_pred_delta
    actual_price_next = base_price + y_test_s.values  # actual next-day price = Price_t + ΔPrice_t

    # Evaluate reconstructed price against actual next-day price
    rmse_signal_price = np.sqrt(mean_squared_error(actual_price_next, predicted_price_from_signal))
    mae_signal_price = mean_absolute_error(actual_price_next, predicted_price_from_signal)
    r2_signal_price = r2_score(actual_price_next, predicted_price_from_signal)

    print("\nSubmission (Signal-only) Model:")
    print(" RMSE (Price):", rmse_sig, "MAE (Price):", mae_sig, "R2 (Price test):", r2_sig_test)
    print(" RMSE (reconstructed price):", rmse_signal_price,
        "MAE (reconstructed price):", mae_signal_price,
        "R2 (reconstructed price):", r2_signal_price)

    # Save submission predictions CSV
    submission_df = pd.DataFrame({
        "Date_t": df.loc[X_test_s.index, "Date"].values,
        "Price_t": base_price,
        "Actual_Price_t+1": actual_price_next,
        "Predicted_Price_t+1": predicted_price_from_signal,
        "Actual_Delta": y_test_s.values,
        "Predicted_Delta": y_pred_delta
    })
    submission_df.to_csv(f"{RESULTS_DIR}/submission_predictions.csv", index=False)
    print("Saved results/submission_predictions.csv")

    # Append to model_summary.txt (keep your existing file write; here we append)
    with open(f"{RESULTS_DIR}/model_summary.txt", "a") as f:
        f.write("\nSUBMISSION MODEL (Signal-only -> reconstructed price)\n")
        f.write("-" * 45 + "\n")
        f.write(f"RMSE (Price, test)         : {rmse_sig:.4f}\n")
        f.write(f"MAE  (Price, test)         : {mae_sig:.4f}\n")
        f.write(f"R2   (Price, test)         : {r2_sig_test:.4f}\n\n")
        f.write(f"RMSE (reconstructed price)  : {rmse_signal_price:.4f}\n")
        f.write(f"MAE  (reconstructed price)  : {mae_signal_price:.4f}\n")
        f.write(f"R2   (reconstructed price)  : {r2_signal_price:.4f}\n")



    # ==================================================
    # RESET DATAFRAME FOR PRICE MODELS (NO LEAKAGE)
    # ==================================================

    df = load_data()
    df = build_features(df)



    # Time-based split (NO SHUFFLING)
    X_train, X_test, y_train, y_test = time_train_test_split(df)

    print("Train X shape:", X_train.shape)
    print("Test X shape:", X_test.shape)
    print("Train y shape:", y_train.shape)
    print("Test y shape:", y_test.shape)

    # -----------------------------
    # 2. Linear Regression (Unscaled)
    # -----------------------------
    lr_model, rmse_lr, mae_lr, r2_lr_train, r2_lr_test = train_linear_regression(
        X_train, y_train, X_test, y_test
    )

    print("\nLinear Regression (Unscaled)")
    print("RMSE:", rmse_lr)
    print("MAE :", mae_lr)
    print("R2 Train:", r2_lr_train)
    print("R2 Test :", r2_lr_test)

    # -----------------------------
    # 3. Linear Regression (Scaled)
    # -----------------------------
    lr_scaled_model, scaler, rmse_lrs, mae_lrs, r2_lrs_train, r2_lrs_test = (
        train_scaled_linear_regression(X_train, y_train, X_test, y_test)
    )

    print("\nLinear Regression (Scaled)")
    print("RMSE:", rmse_lrs)
    print("MAE :", mae_lrs)
    print("R2 Train:", r2_lrs_train)
    print("R2 Test :", r2_lrs_test)

    # -----------------------------
    # 4. Random Forest
    # -----------------------------
    rf_model, rmse_rf, mae_rf, r2_rf_train, r2_rf_test = train_random_forest(
        X_train, y_train, X_test, y_test
    )

    print("\nRandom Forest")
    print("RMSE:", rmse_rf)
    print("MAE :", mae_rf)
    print("R2 Train:", r2_rf_train)
    print("R2 Test :", r2_rf_test)
    # Check which features the model actually cares about
    importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
    print("\nRandom Forest Feature Importances:")
    print(importances.sort_values(ascending=False))

    

    # -----------------------------
    # 5. Gradient Boosting
    # -----------------------------
    gb_model, rmse_gb, mae_gb, r2_gb_train, r2_gb_test = train_gradient_boosting(
        X_train, y_train, X_test, y_test
    )

    print("\nGradient Boosting")
    print("RMSE:", rmse_gb)
    print("MAE :", mae_gb)
    print("R2 Train:", r2_gb_train)
    print("R2 Test :", r2_gb_test)

    

    # =============================
    # Save Results & Visualizations
    # =============================

    import os
    import matplotlib.pyplot as plt

    RESULTS_DIR = "results"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # -----------------------------
    # Predictions from final model
    # -----------------------------
    y_test_pred = lr_model.predict(X_test)

    # =============================
    # FINAL EVALUATION PLOTS
    # =============================

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # --------------------------------------------------
    # 1. PRICE MODEL – Linear Regression
    # --------------------------------------------------
    axes[0].plot(
        y_test.values[:100],
        label="Actual Price",
        marker="o",
        markersize=3
    )

    axes[0].plot(
        y_test_pred[:100],
        label="Predicted Price (LR)",
        marker="x",
        markersize=3
    )

    axes[0].set_title("Price Model (Linear Regression)\nActual vs Predicted")
    axes[0].set_xlabel("Time Index")
    axes[0].set_ylabel("Stock Price")
    axes[0].legend()
    axes[0].grid(alpha=0.3)


# --------------------------------------------------
# 2. SIGNAL MODEL – Reconstructed Price
# --------------------------------------------------


    # -----------------------------
    # Save predictions
    # -----------------------------
predictions_df = pd.DataFrame({
        "Actual_Price": y_test.values,
        "Predicted_Price": y_test_pred,
        
    })

predictions_df.to_csv(f"{RESULTS_DIR}/predictions.csv", index=False)
print("Saved results/predictions.csv")

    # -----------------------------
    # Save model summary
    # -----------------------------
with open(f"{RESULTS_DIR}/model_summary.txt", "w") as f:
        f.write("STOCK PRICE PREDICTION – MODEL SUMMARY\n")
        f.write("=" * 45 + "\n\n")

        f.write("Final Model: Linear Regression\n\n")
        f.write(f"RMSE: {rmse_lr:.4f}\n")
        f.write(f"MAE : {mae_lr:.4f}\n")
        f.write(f"R2 Train: {r2_lr_train:.4f}\n")
        f.write(f"R2 Test : {r2_lr_test:.4f}\n\n")

        f.write("Other Models (Comparison)\n")
        f.write("-" * 45 + "\n")
        f.write(f"Random Forest RMSE: {rmse_rf:.4f}\n")
        f.write(f"Gradient Boosting RMSE: {rmse_gb:.4f}\n")

        # Append to model_summary.txt (keep your existing file write; here we append)
  
        f.write("\nSUBMISSION MODEL (Signal-only -> reconstructed price)\n")
        f.write("-" * 45 + "\n")
        f.write(f"RMSE (Price, test)         : {rmse_sig:.4f}\n")
        f.write(f"MAE  (Price, test)         : {mae_sig:.4f}\n")
        f.write(f"R2   (Price, test)         : {r2_sig_test:.4f}\n\n")
        f.write(f"RMSE (reconstructed price)  : {rmse_signal_price:.4f}\n")
        f.write(f"MAE  (reconstructed price)  : {mae_signal_price:.4f}\n")
        f.write(f"R2   (reconstructed price)  : {r2_signal_price:.4f}\n")


print("Saved results/model_summary.txt")






