# ML Walk-Forward Validation: Official Terminology

**Last Updated:** February 3, 2026

---

## The Complete Parameter List

This document defines **exactly 5 parameters** used in ML walk-forward validation.

### ✅ The 3 Core Parameters (REQUIRED)

These are the **only** parameters you absolutely need for walk-forward validation:

#### 1. `train_size` (Training Window)
- **What it is:** How many historical periods to train on
- **Example:** `train_size = 252` means train on 252 days (1 trading year)
- **Purpose:** Defines the initial training data length
- **Walk-forward behavior:** This window **expands** each iteration (gets longer)
- **Alternative names you might see:**
  - `initial_train_days`
  - `training_window`
  - `lookback_period`

#### 2. `test_size` (Test Window)
- **What it is:** How many future periods to test on each iteration
- **Example:** `test_size = 5` means test on next 5 days (1 trading week)
- **Purpose:** Defines the evaluation window size
- **Walk-forward behavior:** This window stays **constant** each iteration
- **Alternative names you might see:**
  - `test_days`
  - `test_window`
  - `prediction_window`

#### 3. `seq_len` (Sequence Length / Lookback) **[LSTM ONLY]**
- **What it is:** How many past periods each LSTM input sample contains
- **Example:** `seq_len = 60` means each input sees 60 days of history
- **Purpose:** Defines LSTM's "memory window" for temporal patterns
- **Not applicable to:** XGBoost (uses tabular features, not sequences)
- **Alternative names you might see:**
  - `sequence_length`
  - `lookback_window`
  - `time_steps`

---

### ⚙️ The 2 Optional Parameters (ADVANCED)

These are **optional**. Default values work well for most use cases.

#### 4. `step_size` (Walk-Forward Step)
- **What it is:** How far to move the window forward each iteration
- **Example:** `step_size = 5` means move 5 days forward each time
- **Common default:** `step_size = test_size` (no overlap)
- **Purpose:** Controls overlap between validation splits
- **Options:**
  - `step_size = test_size` → No overlap (recommended)
  - `step_size < test_size` → Overlapping tests (more splits)
  - `step_size > test_size` → Gaps between tests (faster training)
- **Alternative names you might see:**
  - `stride`
  - `shift_size`
  - `step`

#### 5. `val_size` (Validation Window)
- **What it is:** Slice of training data reserved for hyperparameter tuning
- **Example:** `val_size = 50` means use last 50 days of training for validation
- **Common default:** `val_size = 0` (no validation split)
- **Purpose:** For hyperparameter tuning **within** each training split
- **When to use:**
  - If you want to tune hyperparameters at each split
  - If you have lots of data (can spare 20% for validation)
- **When to skip:**
  - Using default hyperparameters (current approach)
  - Limited data (use all for training)
- **Alternative names you might see:**
  - `validation_size`
  - `val_window`
  - `holdout_size`

---

## Walk-Forward Validation Illustration

```
Data Timeline: [--------- Total Available Data ---------]

Iteration 1:
    Train: [======train_size======]
    Test:                          [test]
    
Iteration 2:
    Train: [=======train_size+step=======]
    Test:                                 [test]
    
Iteration 3:
    Train: [========train_size+2*step========]
    Test:                                      [test]
```

**Key Points:**
- Training window **expands** (gets longer)
- Test window stays **constant size**
- Move forward by `step_size` each time (default = `test_size`)
- No look-ahead bias (always test on future unseen data)

---

## Parameter Relationships

### LSTM Data Requirements
```
train_size >= seq_len + 100
```
- LSTM needs `seq_len` history for each sample
- Plus buffer for meaningful training

### Recommended Defaults
```python
# Core parameters (adjust based on data frequency)
train_size = 252   # 1 year of daily data
test_size = 5      # 1 week of daily data
seq_len = 60       # 2-3 months lookback (LSTM)

# Optional parameters (use defaults)
step_size = test_size  # No overlap
val_size = 0          # No validation split
```

### Data Frequency Adjustments
| Frequency | train_size | test_size | seq_len |
|-----------|------------|-----------|---------|
| Daily     | 252 days   | 5 days    | 60 days |
| Weekly    | 52 weeks   | 1 week    | 12 weeks|
| Monthly   | 24 months  | 1 month   | 6 months|

---

## Model Hyperparameters (NOT Walk-Forward Parameters)

These are **separate** from walk-forward parameters:

### XGBoost Hyperparameters
- `n_estimators`: Number of boosting rounds (default: 100)
- `max_depth`: Tree depth (default: 3)
- `learning_rate`: Shrinkage rate (default: 0.1)

### LSTM Hyperparameters
- `hidden_units`: LSTM layer size (default: 64)
- `dropout_rate`: Regularization (default: 0.3)
- `epochs`: Training epochs (default: 50)

**Important:** Hyperparameters control **how** the model learns.
Walk-forward parameters control **when/how much** data is used.

---

## Consistency Rules

### Variable Names in Code
```python
# ✅ Use these official names
train_size      # Training window
test_size       # Test window
seq_len         # LSTM sequence length
step_size       # Walk-forward step (optional)
val_size        # Validation window (optional)

# ❌ Avoid these inconsistent aliases
initial_train, initial_train_days
test_period, test_days
lstm_sequence, sequence_length, lookback
```

### UI Labels
```python
# Streamlit sidebar labels
"Training Window (train_size)"
"Test Window (test_size)"
"Sequence Length (seq_len)"  # LSTM only
"Step Size (step_size)"      # Optional
"Validation Window (val_size)"  # Optional
```

---

## FAQ

### Q: Why only 5 parameters? I've seen papers with more.
**A:** This is the **minimal complete set** for walk-forward validation. Other parameters (e.g., refit frequency, purge size) are advanced optimizations.

### Q: Is `seq_len` required?
**A:** Only for LSTM. XGBoost doesn't use sequences.

### Q: Should I always use validation splits?
**A:** No. Use `val_size = 0` (no validation) when:
- Using default hyperparameters
- Data is limited
- Not tuning hyperparameters

### Q: What if I see different names in the code?
**A:** Use this document as the **single source of truth**. All code should migrate to these official names.

### Q: Can I add more parameters later?
**A:** Only if truly necessary (e.g., purge size for high-frequency trading). Document new parameters here first.

---

## Resume/Interview Talking Points

**"I implemented a production-ready ML pipeline with walk-forward validation using 3 core parameters:"**

1. **train_size** (expanding training window)
2. **test_size** (constant test window)
3. **seq_len** (LSTM lookback)

**"I kept it simple and transparent - only 5 total parameters (2 optional). No hidden complexity."**

**Key differentiator:** Most academic papers have 10+ parameters. I focused on the essential 5 with clear documentation.

---

## Implementation Checklist

- [x] Define 5 official parameters
- [x] Create terminology reference document
- [ ] Update all Streamlit UI labels
- [ ] Update all function signatures
- [ ] Update all documentation
- [ ] Add tooltips with official terminology
- [ ] Verify consistent usage across codebase

---

**End of Official Terminology Reference**
