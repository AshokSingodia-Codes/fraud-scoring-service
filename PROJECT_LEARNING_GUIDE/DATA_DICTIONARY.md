# Data Dictionary: IEEE-CIS Fraud Scoring Engine

This document provides a comprehensive data dictionary for the features used across the training, validation, evaluation, and serving pipelines of the fraud scoring engine.

---

## 1. Primary Identifier & Target Variables

| Feature Name | Meaning & Description | Data Type | Role | Origin / Source |
|---|---|---|---|---|
| `TransactionID` | Unique numeric identifier for the transaction event. Used strictly for joining transaction and identity records. | `int32` / `string` | Identifier | Raw Kaggle (`train_transaction.csv` / `train_identity.csv`) |
| `isFraud` | Binary ground-truth fraud label: `1` indicates verified fraud (chargeback, reported unauthorized transaction); `0` indicates legitimate transaction. | `uint8` | Target Variable | Raw Kaggle (`train_transaction.csv`) |
| `TransactionDT` | Timedelta in seconds from an undisclosed reference datetime. Spans from Day 1 (`86400`) to Day 182 (`15811131`). | `int32` | Time Index (used for splitting & derived features) | Raw Kaggle (`train_transaction.csv`) |

---

## 2. Core Transaction & Financial Features

| Feature Name | Meaning & Description | Data Type | Role | Origin / Source |
|---|---|---|---|---|
| `TransactionAmt` | Transaction payment amount in USD. Highly right-skewed (\$0.25 to \$31,937.39). | `float32` | Input Feature | Raw Kaggle (`train_transaction.csv`) |
| `log_TransactionAmt` | Log-transformed transaction amount: $\ln(1 + \text{TransactionAmt})$. Stabilizes extreme variance. | `float32` | Derived Feature | FeatureBuilder (`builder.py`) |
| `amt_cents` | The fractional cents component of the amount: $\text{TransactionAmt} \pmod{1.0}$. Captures pricing patterns (e.g. .99 vs .00). | `float32` | Derived Feature | FeatureBuilder (`builder.py`) |
| `amt_is_round` | Boolean flag indicating whether the transaction amount is a whole round dollar value ($\text{cents} == 0$). Often elevated in card testing attacks. | `float32` | Derived Feature | FeatureBuilder (`builder.py`) |
| `ProductCD` | Product code for the transaction service (categorical with levels: `W`, `C`, `R`, `H`, `S`). | `category` | Input Feature | Raw Kaggle (`train_transaction.csv`) |

---

## 3. Card & Identity Entity Features

| Feature Name | Meaning & Description | Data Type | Role | Origin / Source |
|---|---|---|---|---|
| `card1` | Payment card issuer / Bank Identification Number (BIN) proxy. High cardinality (13,553 unique categories). | `float32` | Input Feature | Raw Kaggle (`train_transaction.csv`) |
| `card2` | Secondary payment card issuer / routing sub-code (approx. 500 unique categories). | `float32` | Input Feature | Raw Kaggle (`train_transaction.csv`) |
| `card3` | Payment card country / region code proxy. | `float32` | Input Feature | Raw Kaggle (`train_transaction.csv`) |
| `card4` | Payment card network brand (categorical: `visa`, `mastercard`, `discover`, `american express`). | `category` | Input Feature | Raw Kaggle (`train_transaction.csv`) |
| `card5` | Payment card category / product type proxy (e.g. platinum, business, standard). | `float32` | Input Feature | Raw Kaggle (`train_transaction.csv`) |
| `card6` | Payment card type (categorical: `debit`, `credit`, `charge card`, `debit or credit`). | `category` | Input Feature | Raw Kaggle (`train_transaction.csv`) |
| `addr1` | Purchaser billing postal / region code proxy (approx. 332 unique categories). | `float32` | Input Feature | Raw Kaggle (`train_transaction.csv`) |
| `addr2` | Purchaser billing country code proxy. | `float32` | Input Feature | Raw Kaggle (`train_transaction.csv`) |
| `dist1` | Geographic distance proxy between billing address, IP address, and merchant location. High missingness (~60%). | `float32` | Input Feature | Raw Kaggle (`train_transaction.csv`) |
| `dist2` | Secondary distance proxy (often shipping address distance). Extreme missingness (~93%). | `float32` | Input Feature | Raw Kaggle (`train_transaction.csv`) |

---

## 4. Derived Temporal & Interaction Features

| Feature Name | Meaning & Description | Data Type | Role | Origin / Source |
|---|---|---|---|---|
| `day_index` | Calendar day index: $\lfloor \text{TransactionDT} / 86400 \rfloor$. Ranges from Day 1 to Day 182. | `float32` | Derived Feature | FeatureBuilder (`builder.py`) |
| `hour_of_day` | Diurnal hour: $\lfloor \text{TransactionDT} / 3600 \rfloor \pmod{24}$. Ranges from 0 to 23. Captures higher nighttime fraud risk. | `float32` | Derived Feature | FeatureBuilder (`builder.py`) |
| `day_of_week` | Day of week index: $\lfloor \text{TransactionDT} / 86400 \rfloor \pmod{7}$. Ranges from 0 to 6. | `float32` | Derived Feature | FeatureBuilder (`builder.py`) |
| `card1_addr1` | Compound string key concatenating card issuer and billing region (`card1_addr1`). Represents unique card-address pairs. | `string` (internal) | Interaction Basis | FeatureBuilder (`builder.py`) |

---

## 5. Email Domain & Matching Features

| Feature Name | Meaning & Description | Data Type | Role | Origin / Source |
|---|---|---|---|---|
| `P_emaildomain` | Purchaser email domain (e.g. `gmail.com`, `yahoo.com`, `anonymous.com`). High cardinality (59 categories). | `category` | Input Feature | Raw Kaggle (`train_transaction.csv`) |
| `R_emaildomain` | Recipient email domain. Used primarily in transfer/escrow products. High missingness (~76%). | `category` | Input Feature | Raw Kaggle (`train_transaction.csv`) |
| `P_emaildomain_provider` | Base email provider extracted from `P_emaildomain` (e.g. `gmail`, `yahoo`, `hotmail`). | `category` | Derived Feature | FeatureBuilder (`builder.py`) |
| `P_emaildomain_suffix` | Domain suffix extracted from `P_emaildomain` (e.g. `com`, `net`, `co.uk`). | `category` | Derived Feature | FeatureBuilder (`builder.py`) |
| `R_emaildomain_provider` | Base email provider extracted from `R_emaildomain`. | `category` | Derived Feature | FeatureBuilder (`builder.py`) |
| `R_emaildomain_suffix` | Domain suffix extracted from `R_emaildomain`. | `category` | Derived Feature | FeatureBuilder (`builder.py`) |
| `email_domain_match` | Boolean flag indicating whether purchaser and recipient email domains match (`1.0` if match, `0.0` otherwise). | `float32` | Derived Feature | FeatureBuilder (`builder.py`) |

---

## 6. Frequency Encodings (`*_fq`)

Each frequency feature represents the occurrence count of a category observed *strictly in the training partition*. Unseen serving categories default to `0.0`.

| Feature Name | Base Feature Encoded | Data Type | Role | Interpretation |
|---|---|---|---|---|
| `card1_fq` | `card1` | `float32` | Derived Feature | Frequency of card issuer. High frequency indicates established institutional issuers; low frequency may indicate rare cards. |
| `card2_fq` | `card2` | `float32` | Derived Feature | Frequency of sub-routing code. |
| `card3_fq` | `card3` | `float32` | Derived Feature | Frequency of card country. |
| `card5_fq` | `card5` | `float32` | Derived Feature | Frequency of card product tier. |
| `addr1_fq` | `addr1` | `float32` | Derived Feature | Frequency of billing postal region. |
| `addr2_fq` | `addr2` | `float32` | Derived Feature | Frequency of billing country. |
| `P_emaildomain_fq` | `P_emaildomain` | `float32` | Derived Feature | Frequency of purchaser email provider. |
| `DeviceInfo_fq` | `DeviceInfo` | `float32` | Derived Feature | Frequency of device hardware profile. Rare devices often correlate with emulators. |
| `id_30_fq` | `id_30` (OS) | `float32` | Derived Feature | Frequency of operating system version. |
| `id_31_fq` | `id_31` (Browser) | `float32` | Derived Feature | Frequency of browser string. |
| `id_33_fq` | `id_33` (Screen) | `float32` | Derived Feature | Frequency of screen resolution. |
| `card1_addr1_fq` | `card1_addr1` | `float32` | Derived Feature | Frequency of joint card-address pair. Unusual combinations yield low frequencies. |

---

## 7. Identity & Device Features

| Feature Name | Meaning & Description | Data Type | Role | Origin / Source |
|---|---|---|---|---|
| `DeviceType` | Type of client device (categorical: `desktop`, `mobile`). Missing for card-present/simple transactions. | `category` | Input Feature | Raw Kaggle (`train_identity.csv`) |
| `DeviceInfo` | Client hardware/OS string (e.g. `Windows`, `iOS Device`, `SM-G960F`, `MacOS`). | `category` | Input Feature | Raw Kaggle (`train_identity.csv`) |
| `id_01` – `id_11` | Anonymized numerical identity metrics (e.g. velocity counters, risk scores, IP proxy scores). | `float32` | Input Features | Raw Kaggle (`train_identity.csv`) |
| `id_12` – `id_38` | Anonymized categorical identity metrics (e.g. browser versions, screen resolution, proxy flags). | `category` | Input Features | Raw Kaggle (`train_identity.csv`) |
| `id_missing_count` | Total count of missing identity fields out of the 38 `id_*` columns. | `float32` | Derived Feature | FeatureBuilder (`builder.py`) |
| `has_identity` | Boolean indicator (`1.0` if any identity attribute is present, `0.0` if non-identity). Identity transactions exhibit a higher fraud rate (~6–8%). | `float32` | Derived Feature | FeatureBuilder (`builder.py`) |

---

## 8. Anonymized Behavioral Feature Blocks

| Block Name | Number of Columns | Meaning & Description | Data Type | Missingness Level |
|---|---|---|---|---|
| `C1` – `C14` | 14 columns | Count aggregations: velocity counts of cards, addresses, emails associated with the transaction over sliding windows. | `float32` | Minimal (< 1%) |
| `D1` – `D15` | 15 columns | Timedelta accumulators: days elapsed since previous transactions on the same card or email. | `float32` | Moderate to High (10% to 80%) |
| `M1` – `M9` | 9 columns | Match indicators: boolean/categorical matches between billing and shipping names, addresses, and cards (`T`, `F`). | `category` | High (40% to 75%) |
| `V1` – `V339` | 339 columns | Vesta engineering blocks: rich engineered transaction velocity, volume, and consistency features across rolling windows. | `float32` | High (15% to 85%) |
