> **DEPRECATED:** This document describes the legacy Streamlit UI. The platform now uses a React + TypeScript frontend. See `frontend/src/pages/MetalsAnalytics.tsx` for the current implementation.

# Streamlit Transparency Display - Summary (Legacy)

**Location:** `frontend/src/pages/MetalsAnalytics.tsx (legacy: apps/pages/2_📊_Metals_Analytics.py)` → "ML Price Prediction"  
**Status:** ✅ Complete transparency display implemented

---

## What's Shown in the Legacy Streamlit App

### 1. Data Preparation Transparency Section
**Location:** Displayed immediately after features are created (before model training)  
**Display:** Expandable section (expanded by default)

#### ✅ What We DID (7 items)
Shown in the app:
1. Forward filled missing data
2. Created log returns (not arithmetic)
3. Mixed expanding and rolling windows
4. All features lagged (no look-ahead)
5. Dropped rows with NaN in features
6. StandardScaler for LSTM only
7. Auto-balanced class weights

#### ❌ What We DID NOT Do (6 items)
Shown in the app:
1. NO outlier removal/capping
2. NO PCA or dimensionality reduction
3. NO Box-Cox transforms
4. NO synthetic data (SMOTE)
5. NO hyperparameter tuning
6. NO ensemble methods

#### ⚠️ To Be Decided (5 items)
Shown in the app:
1. Outlier treatment: Keep / Cap / Winsorize
2. Hyperparameter tuning: Use defaults or optimize?
3. Additional features: Ratios, cross-asset, more indicators?
4. Model selection: Try LightGBM, CatBoost, Transformers?
5. Class imbalance strategy: Adjust threshold or oversample?

### 2. Outlier Analysis
**Location:** Within transparency section  
**Display:** Metrics + interpretation

Shows:
- Total returns count
- Outliers (>3σ) count and percentage
- Min return (%)
- Max return (%)
- **Interpretation** (based on outlier percentage)
- **Action Taken** (explicitly states: "NONE - All data kept")

### 3. Class Distribution Analysis
**Location:** Within transparency section  
**Display:** Metrics + recommendation

Shows:
- Down days count and percentage
- Up days count and percentage
- **Recommendation** (auto class_weight if >65% imbalance)

### 4. Comprehensive Interpretation Guide
**Location:** At the bottom (expandable section)  
**Display:** Detailed explanations

Includes:
- How to interpret accuracy, precision, recall, F1, ROC AUC
- Walk-forward validation explained
- XGBoost vs LSTM comparison
- Feature importance interpretation
- **Full data preparation transparency** (repeated with more detail)
- **Important disclaimers** (not production-ready, no transaction costs)
- **Learn more** (links to documentation)

---

## Display Flow

```
User clicks "Run ML Prediction"
    ↓
Features created
    ↓
✅ Success message: "Features created: X rows, Y features"
    ↓
📋 DATA PREPARATION TRANSPARENCY (expanded by default)
    ├── What We DID (✅ checkmarks)
    ├── What We DID NOT Do (❌ crosses)
    ├── To Be Decided (⚠️ warnings)
    ├── Outlier Analysis (metrics + interpretation)
    └── Class Distribution (metrics + recommendation)
    ↓
Model training begins
    ↓
Results displayed
    ↓
ℹ️ INTERPRETATION GUIDE (expandable, collapsed by default)
    ├── How to interpret metrics
    ├── Walk-forward validation explained
    ├── XGBoost vs LSTM comparison
    ├── Feature importance guide
    ├── 🔍 DATA PREPARATION TRANSPARENCY (detailed version)
    │   ├── ✅ What We DID (7 items with explanations)
    │   ├── ❌ What We DID NOT Do (6 items with reasons)
    │   └── ⚠️ To Be DECIDED (5 items with options)
    ├── ⚠️ IMPORTANT DISCLAIMERS (warning box)
    └── 📚 LEARN MORE (links to docs)
```

---

## Screenshots Guide (What User Will See)

### Section 1: Initial Transparency (Top of Results)
```
✅ Features created: 1234 rows, 15 features

📋 Data Preparation Transparency  [expanded]

### What We Did

#### ✅ Completed:                  #### ❌ NOT Done:
- ✓ Forward filled missing data    - ✗ NO outlier removal/capping
- ✓ Created log returns...          - ✗ NO PCA or dimensionality...
- ✓ Mixed expanding and rolling...  - ✗ NO Box-Cox transforms
- ✓ All features lagged...          - ✗ NO synthetic data (SMOTE)
- ✓ Dropped rows with NaN...        - ✗ NO hyperparameter tuning
                                     - ✗ NO ensemble methods

#### ⚠️ To Be Decided:
- ? Outlier treatment: Keep / Cap / Winsorize
- ? Hyperparameter tuning: Use defaults or optimize?
- ? Additional features: Ratios, cross-asset, more indicators?
- ? Model selection: Try LightGBM, CatBoost, Transformers?
- ? Class imbalance strategy: Adjust threshold or oversample?

---

### 🔍 Outlier Analysis

[Total Returns]  [Outliers (>3σ)]  [Min Return]  [Max Return]
    1234             15 (1.22%)      -8.45%        +7.23%

ℹ️ Interpretation: Low outlier count (<1%) - likely real price 
   movements. Keep all data.
   
   Action Taken: NONE - All data kept. User can decide on 
   capping/winsorizing later.

---

### ⚖️ Class Distribution (Up vs Down Days)

[Down Days (0)]          [Up Days (1)]
612 (49.6%)             622 (50.4%)

✅ No class weighting needed (balanced)
```

### Section 2: Interpretation Guide (Bottom, Expandable)
```
ℹ️ Understanding ML Prediction Results  [click to expand]

[When expanded shows:]

### How to Interpret These Results
[Accuracy benchmarks, precision, recall, F1, ROC AUC explained]

### Walk-Forward Validation
[Why expanding window, why 1-week test periods]

### XGBoost vs LSTM
[When each model wins]

### Feature Importance (XGBoost)
[What it means]

---

### 🔍 Data Preparation Transparency

#### ✅ What We DID:

1. Forward filled missing data
   - Commodities trade continuously
   - Gaps are typically weekends/holidays

2. Used LOG returns (not arithmetic)
   - Time-additive: log(P_t / P_{t-1})
   - Better for ML and time-series
   - More symmetric distribution

[... continues with 7 items, each with explanation ...]

#### ❌ What We DID NOT Do:

1. NO outlier removal/capping
   - All data kept (including extreme returns)
   - We REPORT outliers but don't remove them
   - You decide: Keep / Cap / Winsorize

[... continues with 6 items, each with reason ...]

#### ⚠️ To Be DECIDED (By You):

1. Outlier Treatment
   - Check outlier report above
   - Options: Keep all / Cap at 3σ / Winsorize 1%/99%
   - Current: Keeping all (transparent)

[... continues with 5 items, each with options ...]

---

### ⚠️ Important Disclaimers

⚠️ This is Educational/Research Code:

- ❌ Not production-ready for live trading
- ❌ No transaction costs included
- ❌ No slippage modeling
- ❌ No position sizing or risk management

[... continues with guidance ...]

---

### 📚 Learn More

ℹ️ Documentation:
   - docs/ML_PRICE_PREDICTION.md - Complete technical details
   - docs/ML_TRANSPARENCY_REPORT.md - All data prep decisions
   - docs/ML_QUICK_START.md - Installation and usage guide

   Code:
   - src/data/ml_features.py - Feature engineering
   - src/models/commodity_direction.py - Models and validation
```

---

## Key Features of Transparency Display

### 1. **Prominent Placement**
- Shows BEFORE results (user sees it immediately)
- Expanded by default (can't miss it)
- Repeated in interpretation guide (reinforcement)

### 2. **Clear Visual Hierarchy**
- ✅ Green checkmarks for completed
- ❌ Red crosses for not done
- ⚠️ Yellow warnings for to be decided
- Color-coded sections

### 3. **Actionable Information**
- Not just "what" but "why"
- Options provided for user decisions
- Links to detailed documentation

### 4. **Explicit About Outliers**
- Reports count and percentage
- Provides interpretation
- **Explicitly states: "NONE - All data kept"**
- Gives user options

### 5. **Educational Focus**
- Explains technical terms
- Provides context for decisions
- Links to learn more

### 6. **Honest Disclaimers**
- Clear warning about production use
- Lists what's missing
- Sets realistic expectations

---

## Comparison: Before vs After

### Before Enhancement
- ❌ Transparency only in "Compare Both" mode
- ❌ Limited detail (just lists)
- ❌ No outlier action stated explicitly
- ❌ No disclaimers about production use

### After Enhancement
- ✅ Transparency in ALL modes (Compare Both, XGBoost Only, LSTM Only)
- ✅ Detailed explanations (with reasons and options)
- ✅ Explicit outlier action: "NONE - All data kept"
- ✅ Clear disclaimers and warnings
- ✅ Repeated in interpretation guide
- ✅ Links to documentation

---

## User Experience Flow

1. **User clicks "Run ML Prediction"**
   - Sees transparency section immediately
   - Can review before waiting for training

2. **User reviews transparency**
   - Understands what was done
   - Sees what was NOT done (just as important)
   - Knows what decisions are deferred

3. **User sees outlier analysis**
   - Actual counts and percentages
   - Interpretation provided
   - **Action taken explicitly stated**

4. **User sees results**
   - Model metrics displayed
   - Can click interpretation guide for more detail

5. **User reads interpretation guide**
   - Reinforces transparency
   - Provides context for decisions
   - Warns about production use

---

## Summary

**Transparency is now:**
- ✅ Prominent (shown first, expanded by default)
- ✅ Comprehensive (7 did + 6 didn't + 5 to decide)
- ✅ Explicit (especially about outliers: "NONE - All data kept")
- ✅ Actionable (provides options for user decisions)
- ✅ Educational (explains why, not just what)
- ✅ Honest (clear disclaimers and warnings)
- ✅ Accessible (in app + documentation)

**The user will NEVER miss:**
- What data prep was done
- What was skipped (and why)
- That outliers are kept (not removed)
- What decisions are theirs to make
- That this is educational code

---

**Last Updated:** February 3, 2026  
**Status:** ⚠️ Legacy — transparency display was in the Streamlit app; see frontend for current implementation
