📈 Stock Price Prediction using External Data

Futures First – Machine Learning Assignment

📌 Problem Statement

The objective of this assignment is to predict the next day’s stock price using information derived from an external Data dataset, under the following constraints:

The next day’s stock price is assumed to be primarily influenced by changes in the Data dataset.

All macroeconomic, fundamental, and external market factors are explicitly ignored.

Only the relationship between the provided datasets may be modeled.

🧠 High-Level Approach

This project follows a data-driven, evidence-based pipeline:

Exploratory Data Analysis (EDA) to understand relationships, stability, and regime behavior.

Feature Engineering guided by EDA and Mutual Information (MI), not assumptions.

Two modeling tracks:

A Price Model (baseline, includes minimal price memory)

A Signal-Only Model (strictly adheres to assignment constraints)

Time-based train–test splitting to avoid look-ahead bias.

Model evaluation and visualization with emphasis on interpretability.

📂 Project Structure
stock-price/
│
├── input/
│   ├── Data 2.csv
│   └── StockPrice.csv
│
├── src/
│   └── train.py
│
├── results/
│   ├── final_model_comparison.png
│   ├── model_evaluation.png
│   ├── predictions.csv
│   ├── submission_predictions.csv
│   └── model_summary.txt
│
├── try.ipynb
└── README.md

🔍 Exploratory Data Analysis (EDA)
1. Correlation Analysis

Direct linear correlation between Data and Price was weak and unstable.

Static correlation alone was insufficient for modeling.

2. Regime Shift Detection

A 60-day rolling correlation between:

% change in Data

% change in Price

revealed a clear regime shift around early 2022:

During 2022, Data experienced sharp movements.

Stock Price remained relatively flat or decoupled.

This indicated a breakdown of the historical lead–lag relationship.

📌 Key Insight:
Models trained on earlier regimes struggled to generalize without re-calibration.

🧩 Feature Engineering
✅ Final Feature Set (Used in Models)
1. External Data Dynamics

data_change_lag1

data_change_ma3

data_change_ma7

data_change_std3

These capture momentum, trend, and volatility in the Data signal.

2. Minimal Price Memory (Baseline Only)

price_lag1
price

This acts as a persistence baseline, not as a leaked target.

🧪 Alternative Feature Experiment (Commented Out)

A second build_features() function exists (commented) that includes:

Double Minima / Support Detection

A technical concept where:

Price forms two nearby local bottoms

Followed by a potential upward reversal

📌 This feature was experimented with but excluded because:

It introduced technical assumptions

Did not improve generalization

Was outside strict assignment scope

🤖 Modeling Strategy
1️⃣ Signal-Only Model (Assignment-Compliant)

Inputs: Only Data-derived features

Target: Next-day price change (ΔPrice)

Output: Reconstructed next-day price

This model strictly follows the assignment rule:

“Only the relationship between the provided datasets should be modeled.”

2️⃣ Price Model (Baseline Comparison)

Includes price_lag1

Demonstrates how price memory dominates prediction

Used only for comparison, not submission logic

⏱️ Train–Test Splitting

Time-based split (no shuffling)

Prevents future information leakage

Reflects real-world deployment conditions

📊 Results & Visualizations
Key Outputs:

Actual vs Predicted plots (Linear Regression & Signal Model)

Residual analysis

Model comparison (RMSE)

Prediction CSVs

Model summary report

All results are saved automatically in the results/ directory.

⚠️ Challenges Faced
1. Price Dominance vs Assignment Constraints

Including price_lag1 and price led to near-perfect predictions.

However, this used minimal information from the Data dataset.

Careful separation was required between:

Assignment-compliant model

Baseline comparison model

2. Regime Shift (2022)

The relationship between Data and Price changed significantly.

Models trained on earlier data struggled post-2022.

Highlighted the need for:

Rolling retraining

Adaptive models

3. Lack of Linear Correlation

Traditional correlation analysis failed.

Signal was non-linear and distributed.

Required use of:

Rolling statistics

Mutual Information (MI)

Trend & volatility features

4. Feature Leakage Risks

Accidental reuse of intermediate targets (e.g., ΔPrice)

Required explicit dataframe rebuilding between models

Reinforced the importance of clean pipelines

🚀 Future Improvements

HIT Score / Directional Accuracy

Measure how often the model correctly predicts up/down movement

Rolling Window Training

Adapt models to changing regimes

Regime-Aware Models

Separate models per market regime

Non-linear Models on Signal-Only Data

XGBoost / LightGBM with strict feature control

Probabilistic Forecasting

Predict confidence intervals instead of point estimates

🏁 Final Takeaways

Stock price prediction using external signals is highly regime-dependent

Simple correlations are misleading in financial data

Feature discipline is critical to avoid leakage

Signal-only models are harder — but more realistic and robust

👤 Author

Inha
B.Tech Electrical Engineering
NIT Kurukshetra

