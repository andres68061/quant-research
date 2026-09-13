# ML Feature Engineering - Transparency Report

**Purpose:** Document all data preparation decisions for commodity price prediction ML models.

---

## ✅ Data Preparation COMPLETED

### 1. Missing Data Handling
- **Method:** Forward fill
- **Rationale:** Commodities trade continuously; gaps are typically weekends/holidays
- **Code:** `prices.fillna(method='ffill')`

### 2. Return Calculation
- **Method:** Log returns
- **Formula:** `log(P_t / P_{t-1})`
- **Rationale:** 
  - Time-additive
  - Symmetric
  - Better for ML (more normal distribution)
  - Standard in quantitative finance

### 3. Feature Windows
- **Expanding metrics:**
  - `downside_dev_expanding`: Long-term baseline risk
- **Rolling metrics:**
  - `log_return_1d, 5d, 21d, 63d`: Recent momentum
  - `vol_21d, 63d`: Recent volatility regime
  - `downside_dev_21d, 63d`: Recent downside risk
  - `rsi_14d`: Momentum oscillator
  - `distance_from_ma_50, 200`: Mean reversion signals
- **Rationale:** Mix captures both long-term context and recent regime

### 4. Look-Ahead Bias Prevention
- **Method:** All features lagged by 1 day
- **Example:** `log_return_1d = log_returns.shift(1)`
- **Rationale:** Ensures only past data used for prediction

### 5. Feature Matrix Cleaning
- **Method:** Drop rows with ANY NaN in features
- **Typical loss:** First ~200 rows (for 200-day MA)
- **Rationale:** Models require complete feature matrix

---

## ❌ Data Preparation NOT Done

### 1. Outlier Removal/Capping
- **What we could do:**
  - Cap returns at ±3σ
  - Winsorize at 1%/99% percentiles
  - Remove days with |z-score| > 3
- **Why we didn't:**
  - Outliers may be real events (COVID crash, oil shock)
  - Important learning signals for model
  - User should decide based on their data
- **Transparency:** We REPORT outliers but keep them
- **Check function:** `check_outliers()` in `ml_features.py`

### 2. Scaling/Normalization (for XGBoost)
- **What we could do:** StandardScaler, MinMaxScaler, RobustScaler
- **Why we didn't:** Tree-based models (XGBoost) don't need scaling
- **Note:** LSTM DOES get StandardScaler (applied separately)

### 3. Dimensionality Reduction
- **What we could do:** PCA, t-SNE, UMAP
- **Why we didn't:**
  - Only ~15 features (manageable)
  - Want interpretable features
  - PCA loses interpretability
- **When to consider:** If adding 50+ features

### 4. Complex Transforms
- **What we could do:**
  - Box-Cox transform
  - Yeo-Johnson transform
  - Wavelet decomposition
  - Fourier transforms
- **Why we didn't:**
  - Log returns already approximately normal
  - Adds complexity without clear benefit
  - Harder to interpret

### 5. Synthetic Data Generation
- **What we could do:** SMOTE (Synthetic Minority Over-sampling)
- **Why we didn't:**
  - Class imbalance handled by class weights
  - SMOTE can create unrealistic samples
  - Not recommended for time series
- **Alternative:** Use `class_weight='balanced'` in models

### 6. Feature Selection
- **What we could do:**
  - Remove low-importance features (from XGBoost)
  - Correlation-based removal
  - Recursive feature elimination
- **Why we didn't:**
  - All features theoretically useful
  - XGBoost handles redundancy well
- **When to do:** If overfitting or want faster training

---

## ⚠️ Decisions TO BE MADE BY USER

### 1. Outlier Treatment

**Check the outlier report:**
```python
outliers = metadata['outliers']
print(f"Outliers: {outliers['outlier_count']} ({outliers['outlier_pct']:.2f}%)")
print(f"Interpretation: {outliers['interpretation']}")
```

**Options:**

| Method | Code | Use When |
|--------|------|----------|
| **Keep all** | (default) | <1% outliers, want to learn from extremes |
| **Cap at 3σ** | `returns.clip(-3*std, 3*std)` | 1-3% outliers, reduce extreme predictions |
| **Winsorize** | `from scipy.stats import mstats; mstats.winsorize(returns, limits=[0.01, 0.01])` | >3% outliers, keep distribution shape |

### 2. Class Imbalance Handling

**Check class distribution:**
```python
dist = metadata['class_distribution']
print(f"Down days: {dist['class_0_pct']:.1f}%")
print(f"Up days: {dist['class_1_pct']:.1f}%")
print(f"Recommendation: {dist['recommendation']}")
```

**Current strategy:**
- Auto-applies `class_weight='balanced'` if >65% imbalance

**Alternative strategies:**
- SMOTE (not recommended for time series)
- Adjust classification threshold (instead of 0.5, try 0.4 or 0.6)
- Oversample minority class manually

### 3. Hyperparameter Tuning

**Current defaults:**

**XGBoost:**
```python
n_estimators=100
max_depth=3          # Shallow to prevent overfitting
learning_rate=0.1
```

**LSTM:**
```python
sequence_length=60
hidden_units=64
dropout_rate=0.3
learning_rate=0.001
epochs=50
```

**To tune:**
- Use grid search or random search
- Be careful: Easy to overfit on walk-forward splits
- Recommendation: Only tune if baseline performance poor

### 4. Additional Features

**Easy to add:**
- More lag periods (2d, 3d returns)
- More MA distances (10d, 20d, 100d)
- Additional technical indicators (MACD, Bollinger Bands, ATR)

**Medium complexity:**
- Ratio features (Gold/Silver, Copper/Gold)
- Cross-asset features (SPY returns, VIX, DXY)
- Rolling z-scores

**Hard:**
- Sentiment indicators (news, social media)
- Order flow metrics (volume, bid-ask)
- Macroeconomic features (interest rates, inflation)

### 5. Model Selection

**Current approach:** Compare XGBoost vs LSTM

**Alternatives:**
- LightGBM (faster than XGBoost)
- CatBoost (handles categoricals well)
- Random Forest (ensemble of trees)
- Linear models (logistic regression, SGD)
- Neural networks (MLP, GRU, Transformer)

**Ensemble methods:**
- Stacking (use both XGBoost and LSTM predictions)
- Blending (weighted average)
- Voting classifier

---

## Code Examples

### Example 1: Check Outliers Before Training

```python
from src.data.ml_features import create_ml_features_with_transparency

features_df, metadata = create_ml_features_with_transparency(
    prices['GLD'],
    symbol='GLD'
)

# Check outliers
outliers = metadata['outliers']
print(f"\nOutlier Analysis:")
print(f"  Total returns: {outliers['total_returns']}")
print(f"  Outliers (>3σ): {outliers['outlier_count']} ({outliers['outlier_pct']:.2f}%)")
print(f"  Min return: {outliers['min_return']:.2f}%")
print(f"  Max return: {outliers['max_return']:.2f}%")
print(f"  Interpretation: {outliers['interpretation']}")
print(f"  Action: {outliers['action_taken']}")

# If you want to see outlier dates
if outliers['outlier_count'] > 0:
    print("\nOutlier dates:")
    for date in outliers['outlier_dates'][:5]:  # Show first 5
        print(f"  {date}")
```

### Example 2: Cap Outliers (If You Decide To)

```python
import numpy as np

def cap_outliers(returns, n_std=3):
    """Cap returns at ±n_std standard deviations."""
    mean = returns.mean()
    std = returns.std()
    lower = mean - n_std * std
    upper = mean + n_std * std
    return returns.clip(lower, upper)

# Apply before creating features
log_returns = np.log(prices / prices.shift(1))
log_returns_capped = cap_outliers(log_returns, n_std=3)

# Then create features with capped returns
# (Note: You'd need to modify ml_features.py to accept returns directly)
```

### Example 3: Adjust Class Imbalance Threshold

```python
# In your XGBoost model, adjust threshold:
y_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1

# Instead of default 0.5:
threshold = 0.55  # Require higher confidence for "up" prediction
y_pred = (y_proba > threshold).astype(int)

# Evaluate
accuracy = (y_pred == y_test).mean()
```

### Example 4: Add Custom Features

```python
from src.data.ml_features import create_ml_features

# Create standard features
df = create_ml_features(prices, symbol='GLD')

# Add custom features
df['momentum_5_20'] = df['log_return_5d'] - df['log_return_21d']
df['volatility_ratio'] = df['vol_21d'] / df['vol_63d']
df['extreme_returns'] = (abs(df['log_return_1d']) > df['vol_21d'] / np.sqrt(252)).astype(int)

# Drop NaNs
df = df.dropna()
```

---

## Transparency Checklist

When using this ML implementation, you should:

- [x] **Understand feature creation:** All features lagged, no look-ahead
- [x] **Know outlier status:** Kept in data, not removed
- [x] **Understand scaling:** XGBoost no scaling, LSTM uses StandardScaler
- [x] **Check class balance:** Auto-handled if >65% imbalance
- [x] **Review validation:** Walk-forward expanding window
- [ ] **Decide on outliers:** Keep / Cap / Winsorize based on your data
- [ ] **Tune hyperparameters:** If baseline performance unsatisfactory
- [ ] **Add features:** If needed for your specific use case
- [ ] **Consider ensembles:** If single model performance plateaus

---

## Summary Table

| Step | Status | Action Taken | User Decision |
|------|--------|--------------|---------------|
| Missing data | ✅ Done | Forward fill | - |
| Returns | ✅ Done | Log returns | - |
| Features | ✅ Done | 15 features (lagged) | Can add more |
| Outliers | ❌ Not done | Kept all, reported | Cap/Winsorize? |
| Scaling (XGBoost) | ❌ Not done | Not needed | - |
| Scaling (LSTM) | ✅ Done | StandardScaler | - |
| Class balance | ✅ Done | Auto class_weight | Could use SMOTE |
| Hyperparameters | ❌ Not done | Using defaults | Tune if needed |
| Feature selection | ❌ Not done | Keep all | Remove if overfitting |
| Validation | ✅ Done | Walk-forward expanding | - |

---

**Last Updated:** February 3, 2026  
**Maintained By:** Quantamental Research Platform Team
