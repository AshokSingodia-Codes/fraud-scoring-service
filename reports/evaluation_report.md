# Evaluation & Performance Report

## 1. Summary of Model Performance

- **Validation PR-AUC:** 0.6136
- **Test PR-AUC:** 0.5110
- **Test ROC-AUC:** 0.9018
- **Test ECE (Before Cal):** 0.0129 -> **(After Cal):** 0.0048

## 2. Decision Policy & Cost Optimization

- **Review Threshold (`t_review`):** 0.0496
- **Block Threshold (`t_block`):** 0.7692
- **Cost Assumptions:** FP Review Cost = $10.0, FN Cost = Transaction Amount

## 3. Drift Analysis (PSI)

- **Score PSI (Val vs Test):** 0.0016
### Top Feature PSI Values:

- `V258`: 0.0194
- `C14`: 0.007
- `DeviceInfo`: 0.0
- `card1_fq`: 0.0045
- `C11`: 0.0131
- `day_index`: 11.5141 ⚠️ HIGH DRIFT (>0.2)
- `log_TransactionAmt`: 0.0048
- `C1`: 0.0127
- `D2`: 0.0203
- `card1`: 0.004
- `addr1`: 0.0035
- `card2`: 0.0101
- `C13`: 0.0248
- `card1_addr1_fq`: 0.0066
- `C4`: 0.0102
- `D10`: 0.0342
- `addr1_fq`: 0.0048
- `P_emaildomain`: 0.0
- `id_31`: 0.0
- `card2_fq`: 0.0025
