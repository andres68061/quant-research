# ML Parameter Terminology - Official Reference

**Last Updated:** February 3, 2026  
**Status:** ✅ Locked - No more parameter additions without explicit user request

---

## 🎯 The 3 Core Parameters (REQUIRED)

These are the ONLY parameters you need to configure for walk-forward validation:

### 1. `train_size` (Training Window Length)
**What it is:** How many past periods you use to train the model.

**Examples:**
- Daily: 252 (1 year), 504 (2 years), 1260 (5 years)
- Weekly: 52 (1 year), 104 (2 years), 260 (5 years)
- Monthly: 12 (1 year), 24 (2 years), 60 (5 years)

**Terminology:**
- ✅ **Official:** `train_size` or "training window length"
- ❌ **Avoid:** "initial training days", "training period", "lookback period"

### 2. `test_size` (Test Window Length)
**What it is:** How many future unseen periods you evaluate each round.

**Examples:**
- Daily: 5 (1 week), 21 (1 month)
- Weekly: 1 (1 week), 4 (1 month)
- Monthly: 1 (1 month), 3 (1 quarter)

**Terminology:**
- ✅ **Official:** `test_size` or "test window length"
- ❌ **Avoid:** "test period", "evaluation period", "forecast horizon"

### 3. `seq_len` (Sequence Length - LSTM Only)
**What it is:** How many past periods each LSTM input sample contains (lookback).

**Examples:**
- Daily: 60 (3 months), 126 (6 months), 252 (1 year)
- Weekly: 52 (1 year), 104 (2 years)
- Monthly: 12 (1 year), 24 (2 years), 36 (3 years)

**Terminology:**
- ✅ **Official:** `seq_len`, "sequence length", or "lookback"
- ❌ **Avoid:** "window size", "history length", "memory"

**Note:** XGBoost doesn't use sequences, so this is N/A for XGBoost.

---

## 🔧 Optional Advanced Parameters (Hidden by Default)

### 4. `step_size` (Walk-Forward Step Size)
**What it is:** How far you move the window forward each iteration.

**Default:** `step_size = test_size` (no overlap between test periods)

**Examples:**
- If `test_size = 5`, then `step_size = 5` (standard)
- If `test_size = 5`, `step_size = 1` (overlapping, 5x more splits)

**When to change:**
- Want more validation splits (use `step_size < test_size`)
- Want faster training (use `step_size > test_size`)
- **Recommendation:** Leave at default unless you have specific reason

**Terminology:**
- ✅ **Official:** `step_size` or "walk-forward step"
- ❌ **Avoid:** "stride", "shift", "increment"

### 5. Hyperparameters (Model-Specific Tuning)

**XGBoost:**
- `n_estimators`: Number of trees (default 100)
- `max_depth`: Tree depth (default 3)
- `learning_rate`: Step size (default 0.1)

**LSTM:**
- `hidden_units`: Layer size (default 64)
- `dropout_rate`: Regularization (default 0.3)
- `epochs`: Training iterations (default 50)

**When to change:**
- Poor baseline performance
- Want to tune for better accuracy
- Specific domain knowledge

**Recommendation:** Use defaults first, only tune if needed.

---

## 📊 Visual Summary

```
WALK-FORWARD VALIDATION (Expanding Window)

Timeline: [-----------------------------------------------]
          Day 0                                    Day 5000

Split 1:  [train_size=252]→[test_size=5]
          [0-------------252][253-258]

Split 2:  [train_size=257 (expanded)]→[test_size=5]
          [0--------------------257][258-263]

Split 3:  [train_size=262 (expanded)]→[test_size=5]
          [0-------------------------262][263-268]

step_size = test_size = 5 (no overlap)

For LSTM:
Each training sample uses seq_len=60 past days
```

---

## 🎓 Academic vs Our Terminology

| Academic Term | Our Term | Notes |
|---------------|----------|-------|
| Training set size | `train_size` | Window length, not fixed set |
| Test set size | `test_size` | Rolling window |
| Lookback period | `seq_len` | LSTM only |
| Retraining frequency | `step_size` | How often to move forward |
| Expanding window | ✓ (default) | train_size grows each split |

---

## ❌ What We DON'T Have (And Won't Add Without Request)

1. **Rolling window training** (we use expanding)
2. **Validation set within training** (for hyperparameter tuning during training)
3. **Ensemble parameters** (combining models)
4. **Feature selection threshold** (we use all features)
5. **Outlier capping threshold** (we keep all outliers)

If you need any of these → Explicit request required!

---

## ✅ Parameter Count Guarantee

**Basic usage:** 3 parameters (train_size, test_size, seq_len)

**Advanced usage:** 3 core + 1 step_size + optional hyperparameters

**We will NOT add more core parameters without explicit discussion!**

---

## 📝 Quick Reference Card

```yaml
QUICK SETUP (Default - Works for Most Cases):

Daily Data:
  train_size: 252      # 1 year
  test_size: 5         # 1 week
  seq_len: 60          # 3 months (LSTM)
  step_size: 5         # = test_size (default)

Weekly Data:
  train_size: 104      # 2 years
  test_size: 1         # 1 week
  seq_len: 52          # 1 year (LSTM)
  step_size: 1         # = test_size (default)

Monthly Data:
  train_size: 60       # 5 years
  test_size: 1         # 1 month
  seq_len: 24          # 2 years (LSTM)
  step_size: 1         # = test_size (default)
```

---

## 🚫 Terminology to AVOID

Don't say:
- ❌ "Initial training period"
- ❌ "Test period length"
- ❌ "LSTM lookback window"
- ❌ "Validation window"
- ❌ "Rebalancing frequency"

Always say:
- ✅ `train_size`
- ✅ `test_size`
- ✅ `seq_len` (LSTM only)
- ✅ `step_size` (optional)

---

## Summary

**You asked:** "how many things are there that we need to set?"

**Answer:** **EXACTLY 3 core parameters:**
1. `train_size`
2. `test_size`
3. `seq_len` (LSTM only)

**Everything else is optional and hidden in "Advanced Settings".**

**This will NOT change unless you explicitly request it!**

---

**Maintained by:** Quantamental Research Platform  
**Locked:** February 3, 2026  
**No more parameter creep!** ✅
