# Invoice / Quote Anomaly Detector (Rules + ML Triage)

Upload a CSV → auto-detect suspicious invoices/quotes → export a scored output.

This project combines:
- **Business rules** (GST mismatch, total mismatch, overpricing, duplicate)
- **ML-based anomaly score** (Isolation Forest) for “triage” + **Needs Review** bucket

> ⚠️ Important: Do **not** upload sensitive/client data to a public repo.

---

## Demo (Screenshots)

- Home / Mapping: `docs/app_home.png`
- Results table: `docs/results_table.png`

---

## What problem does this solve?

Finance / procurement teams waste time manually checking:
- Wrong totals
- GST mistakes
- Overpriced lines
- Duplicate invoices/quotes

This app helps you:
- **Quickly flag** high-risk rows
- Put uncertain cases in **Needs Review**
- Keep the rest as **OK**
- Export a clean, scored CSV

---

## Output labels (FLAGGED / NEEDS_REVIEW / OK)

Your results show 3 buckets:

### ✅ OK
No strong issues found.

### ⚠️ NEEDS_REVIEW
Not clearly wrong, but the ML risk score / price deviation looks suspicious.
Good for “human check”.

### 🚩 FLAGGED
Strong signals (rules) indicate likely anomaly (ex: GST mismatch, total mismatch, duplicate, overprice).
High priority.

---

## How it works (simple)

### Step 1: Column mapping
Client files can have different column names (e.g., `invoice_no` instead of `doc_id`).
In the app, you map each required field to the correct column.

### Step 2: Rules engine
The app calculates:
- **expected_gst** = `subtotal * gst_rate`
- **expected_total** = `subtotal + gst_amount`

Rules:
- **gst_mismatch_rule** → if gst_amount doesn’t match expected_gst
- **total_mismatch_rule** → if total doesn’t match expected_total
- **duplicate_rule** → repeated document id + item + amounts
- **overprice_rule** → unit_price way higher than vendor/item baseline

### Step 3: ML Triage (Isolation Forest)
Model assigns an anomaly score based on numeric patterns.
It helps detect “weird” pricing patterns even if rules don’t fire.

### Step 4: Final decision
- If **any rule** triggers → `final_status = FLAGGED`
- Else if ML score is high and/or price deviation is high → `final_status = NEEDS_REVIEW`
- Else → `final_status = OK`

---

## Required columns (minimum)

Your CSV must contain these fields (names can differ; you map them in the app):

- `doc_id` (or invoice_no / quote_no)
- `doc_date` (YYYY-MM-DD recommended)
- `vendor`
- `city` (optional but useful)
- `payment_terms_days` (optional)
- `item`
- `category` (optional)
- `qty`
- `unit_price`
- `subtotal`
- `gst_rate`
- `gst_amount`
- `total`

### Date format
Recommended: `YYYY-MM-DD`  
Example: `2025-02-02`

---

## Detection settings (side panel explained)

Typical settings you’ll see:

- **GST tolerance**
  - Allows small rounding differences (e.g., ±1 PKR)
- **Total tolerance**
  - Same idea for total rounding
- **Overprice threshold**
  - How aggressive to flag expensive items
- **ML risk threshold**
  - Higher = fewer false alarms, but may miss some anomalies
- **Needs Review threshold**
  - Middle bucket for manual check

**Tip:** Start strict (reduce false positives), then tune later based on client feedback.

---

## Run locally

### 1) Install dependencies
```bash
py -m pip install --user -r requirements.txt
