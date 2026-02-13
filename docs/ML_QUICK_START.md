# ML Price Prediction - Quick Start Guide

**New Feature:** Machine Learning price direction prediction for commodities

---

## Installation

### 1. Install Required Packages

```bash
pip install xgboost tensorflow
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

### 2. Verify Installation

```python
import xgboost
import tensorflow as tf

print(f"XGBoost version: {xgboost.__version__}")
print(f"TensorFlow version: {tf.__version__}")
```

Expected output:
```
XGBoost version: 2.0.x
TensorFlow version: 2.15.x
```

---

## Quick Start - Using Streamlit App

### 1. Run the App

```bash
streamlit run apps/portfolio_simulator.py
```

### 2. Navigate to "Metals Analytics" Page

Click "📊 Metals Analytics" in the sidebar.

### 3. Select ONE Commodity

In the sidebar:
- **Commodities:** Select exactly one (e.g., Gold - GLD)
- **Date Range:** 2+ years recommended (e.g., 2020-01-01 to 2024-12-31)

### 4. Choose ML Price Prediction

In the main area:
- **Analysis Type:** Select "ML Price Prediction"

### 5. Configure ML Settings

In the sidebar (ML Configuration section):
- **Model:** "Compare Both" (XGBoost vs LSTM)
- **Initial Training Days:** 63 (3 months)
- **Test Period Days:** 5 (1 week)

### 6. Run Prediction

Click **"🚀 Run ML Prediction"** in the sidebar.

Wait 2-3 minutes for both models to train.

### 7. Review Results

You'll see:
- ✅ **Model Comparison**: XGBoost vs LSTM accuracy
- ✅ **Confusion Matrices**: True/false positives/negatives
- ✅ **Accuracy Over Time**: Performance across walk-forward splits
- ✅ **Feature Importance**: Which features matter most (XGBoost)
- ✅ **Transparency Report**: What we did and didn't do with data

---

## Quick Start - Using Python API

### Example 1: Compare Models

```python
from pathlib import Path
import pandas as pd
from src.data.ml_features import create_ml_features_with_transparency
from src.models.commodity_direction import compare_models

# Load data
prices_df = pd.read_parquet("data/factors/commodities_prices.parquet")

# Create features for Gold
features_df, metadata = create_ml_features_with_transparency(
    prices_df['GLD'],
    symbol='GLD'
)

print(f"Features created: {metadata['final_rows']} rows, {metadata['total_features']} features")

# Compare XGBoost vs LSTM
results = compare_models(
    features_df,
    initial_train_days=63,  # 3 months
    test_days=5,            # 1 week
    verbose=True
)

# Print results
print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
print(f"XGBoost Accuracy: {results['xgboost']['overall_metrics']['accuracy']:.2%}")
print(f"LSTM Accuracy:    {results['lstm']['overall_metrics']['accuracy']:.2%}")
print(f"Winner: {results['winner'].upper()}")
```

### Example 2: Run Single Model (Faster)

```python
from src.models.commodity_direction import run_walk_forward_validation

# Run only XGBoost (faster)
results = run_walk_forward_validation(
    features_df,
    model_type="xgboost",
    initial_train_days=63,
    test_days=5,
    verbose=True
)

# Print metrics
metrics = results['overall_metrics']
print(f"Accuracy:  {metrics['accuracy']:.2%}")
print(f"Precision: {metrics['precision']:.2%}")
print(f"Recall:    {metrics['recall']:.2%}")
print(f"F1 Score:  {metrics['f1_score']:.2%}")
```

### Example 3: Check Data Transparency

```python
# Create features with transparency
features_df, metadata = create_ml_features_with_transparency(
    prices_df['GLD'],
    symbol='GLD'
)

# Check outliers
print("\n📊 OUTLIER ANALYSIS")
print(f"{'='*60}")
outliers = metadata['outliers']
print(f"Total returns: {outliers['total_returns']}")
print(f"Outliers (>3σ): {outliers['outlier_count']} ({outliers['outlier_pct']:.2f}%)")
print(f"Min return: {outliers['min_return']:.2f}%")
print(f"Max return: {outliers['max_return']:.2f}%")
print(f"Interpretation: {outliers['interpretation']}")
print(f"Action taken: {outliers['action_taken']}")

# Check class balance
print("\n⚖️ CLASS DISTRIBUTION")
print(f"{'='*60}")
dist = metadata['class_distribution']
print(f"Down days (0): {dist['class_0_count']} ({dist['class_0_pct']:.1f}%)")
print(f"Up days (1):   {dist['class_1_count']} ({dist['class_1_pct']:.1f}%)")
print(f"Recommendation: {dist['recommendation']}")

# Check transparency
print("\n✅ WHAT WE DID:")
for item in metadata['transparency']['data_prep_completed']:
    print(f"  ✓ {item}")

print("\n❌ WHAT WE DID NOT DO:")
for item in metadata['transparency']['data_prep_NOT_done']:
    print(f"  ✗ {item}")

print("\n⚠️ TO BE DECIDED:")
for item in metadata['transparency']['to_be_decided']:
    print(f"  ? {item}")
```

---

## Understanding the Results

### Accuracy Benchmarks

| Accuracy | Interpretation | What It Means |
|----------|----------------|---------------|
| 50% | Random baseline | Coin flip |
| 52-55% | Weak signal | Marginally profitable |
| 55-60% | Decent signal | Potentially profitable |
| 60%+ | Strong signal | Excellent if consistent |

### Key Metrics

- **Accuracy**: Overall correct predictions
- **Precision**: When predict UP, how often correct?
- **Recall**: Of all UP days, how many caught?
- **F1 Score**: Balanced metric (harmonic mean)
- **ROC AUC**: 0.5=random, 0.7=good, 0.8+=excellent

### Confusion Matrix

```
                Predicted Down    Predicted Up
Actual Down         TN ✅            FP ❌
Actual Up           FN ❌            TP ✅
```

**Goal:** Maximize TN and TP, minimize FP and FN

---

## Troubleshooting

### Error: "Insufficient data for ML training"

**Problem:** Less than 100 days of data.

**Solution:** Select a longer date range (recommend 2+ years).

### Error: "ModuleNotFoundError: No module named 'xgboost'"

**Problem:** XGBoost not installed.

**Solution:**
```bash
pip install xgboost
```

### Error: "ModuleNotFoundError: No module named 'tensorflow'"

**Problem:** TensorFlow not installed.

**Solution:**
```bash
pip install tensorflow
```

### Warning: "Only one class present - cannot train classifier"

**Problem:** All returns are positive or all negative in selected date range.

**Solution:** Select a longer date range with more diverse market conditions.

### Low Accuracy (<48%)

**Possible reasons:**
1. Insufficient data (need 252+ days for reliable results)
2. Unusual market regime (COVID crash, etc.)
3. Commodity has weak predictability
4. Features need tuning

**Try:**
1. Select longer date range
2. Check outlier analysis (extreme events?)
3. Try different commodity
4. See `docs/ML_PRICE_PREDICTION.md` for tuning advice

---

## What's Next?

### Learn More

- 📚 **Full Documentation:** `docs/ML_PRICE_PREDICTION.md`
- 🔍 **Transparency Report:** `docs/ML_TRANSPARENCY_REPORT.md`
- 🎓 **Code Examples:** See `__main__` blocks in:
  - `src/data/ml_features.py`
  - `src/models/commodity_direction.py`

### Customize

1. **Add features:** Edit `create_ml_features()` in `src/data/ml_features.py`
2. **Tune hyperparameters:** Edit model initialization in `src/models/commodity_direction.py`
3. **Try different validation:** Modify `WalkForwardValidator` parameters
4. **Add new models:** Extend `src/models/commodity_direction.py`

### Experiment

- Try different commodities (Silver, Oil, Copper)
- Compare different time periods (bull vs bear markets)
- Adjust initial training period (30, 63, 126, 252 days)
- Adjust test period (1, 5, 10, 21 days)
- Try different models (add LightGBM, CatBoost, etc.)

---

## FAQ

**Q: How long does it take to train?**

A: Typical timings:
- XGBoost: 10-30 seconds
- LSTM: 1-3 minutes
- Compare Both: 2-4 minutes

**Q: Can I use this for live trading?**

A: ⚠️ **Use with caution.** This is educational code. No transaction costs, slippage, or risk management included. Always paper trade first.

**Q: Why keep outliers?**

A: Transparency. Outliers may be real events (COVID crash). We report them, you decide whether to cap/winsorize.

**Q: Why expanding window instead of rolling?**

A: More realistic. In production, you'd use all past data. Also gives growing training set which improves accuracy.

**Q: Why is XGBoost often better than LSTM?**

A: For commodities with limited data (<1000 samples), XGBoost usually wins. LSTM needs longer sequences (1000+ samples) to shine.

**Q: How do I know if my model is good?**

A: Check:
1. Accuracy > 50% (beats random)
2. Accuracy > 52-55% (profitable potential)
3. Consistent across splits (not just lucky)
4. ROC AUC > 0.6 (decent probability calibration)

---

## Getting Help

- **Documentation:** `docs/ML_PRICE_PREDICTION.md`
- **Transparency:** `docs/ML_TRANSPARENCY_REPORT.md`
- **Code:** `src/data/ml_features.py`, `src/models/commodity_direction.py`
- **Example:** Run module `__main__` blocks to see example usage

---

**Last Updated:** February 3, 2026  
**Version:** 1.0.0
