# app.py
import io
import os
import re
import difflib
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest


# ---------------------------
# Page config + UI styling
# ---------------------------
APP_OWNER = "MoinFarid"
APP_TAG = "MoinFarid"  # used in filenames

st.set_page_config(
    page_title="Invoice / Quote Anomaly Detector",
    page_icon="🧾",
    layout="wide",
)

CUSTOM_CSS = """
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

.hero {
  padding: 18px 18px;
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(20,140,255,0.15), rgba(0,0,0,0.02));
  border: 1px solid rgba(0,0,0,0.07);
  margin-bottom: 14px;
}
.hero h1 { margin: 0; font-size: 2.1rem; }
.hero p { margin: 6px 0 0; color: rgba(0,0,0,0.65); }

.badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 0.85rem;
  border: 1px solid rgba(0,0,0,0.12);
  background: rgba(255,255,255,0.65);
  margin-right: 6px;
}

.card {
  padding: 14px 14px;
  border-radius: 16px;
  border: 1px solid rgba(0,0,0,0.08);
  background: rgba(255,255,255,0.65);
}

[data-testid="stDataFrame"] {
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid rgba(0,0,0,0.08);
}

section[data-testid="stSidebar"] { border-right: 1px solid rgba(0,0,0,0.08); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------
# Helpers
# ---------------------------
def _norm_col(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


ALIASES = {
    "doc_id": ["docid", "doc_id", "invoice", "invoiceid", "invoice#", "invoiceno", "invoice_number", "invoice_no",
               "quote", "quote#", "quoteno", "quote_no", "document", "documentid", "refno", "ref_no"],
    "doc_date": ["docdate", "doc_date", "date", "invoice_date", "invoicedate", "quote_date", "createddate"],
    "vendor": ["vendor", "supplier", "party", "company", "vendorname", "suppliername"],
    "city": ["city", "location", "branch", "site"],
    "payment_terms_days": ["paymentterms", "payment_terms", "terms", "payment_terms_days", "termsdays", "creditdays"],
    "item": ["item", "description", "product", "service", "itemname", "item_desc", "particulars"],
    "category": ["category", "type", "group", "itemcategory"],
    "qty": ["qty", "quantity", "qnty", "units", "unitqty"],
    "unit_price": ["unitprice", "unit_price", "price", "rate", "unitrate", "unitcost", "cost"],
    "subtotal": ["subtotal", "sub_total", "amount", "baseamount", "taxableamount", "netamount"],
    "gst_rate": ["gstrate", "gst_rate", "taxrate", "vat_rate", "sales_tax_rate", "st_rate", "tax_percent"],
    "gst_amount": ["gstamount", "gst_amount", "taxamount", "vat_amount", "sales_tax", "sales_tax_amount", "st_amount"],
    "total": ["total", "grandtotal", "grand_total", "totalamount", "gross", "invoice_total"],
}

SCHEMA = [
    ("doc_id", True, "Invoice/Quote ID"),
    ("doc_date", True, "Date"),
    ("vendor", True, "Vendor/Supplier"),
    ("city", False, "City"),
    ("payment_terms_days", False, "Payment terms (days)"),
    ("item", True, "Item/Description"),
    ("category", False, "Category"),
    ("qty", True, "Quantity"),
    ("unit_price", True, "Unit price"),
    ("subtotal", True, "Subtotal"),
    ("gst_rate", True, "GST rate"),
    ("gst_amount", True, "GST amount"),
    ("total", True, "Total"),
]


def suggest_mapping(columns):
    col_norm = {_norm_col(c): c for c in columns}
    mapping = {}
    for key, _, _ in SCHEMA:
        best = None

        for a in ALIASES.get(key, []):
            a_norm = _norm_col(a)
            if a_norm in col_norm:
                best = col_norm[a_norm]
                break

        if best is None:
            cand = []
            for c in columns:
                c_norm = _norm_col(c)
                cand.append((c, difflib.SequenceMatcher(None, _norm_col(key), c_norm).ratio()))
                for a in ALIASES.get(key, []):
                    cand.append((c, difflib.SequenceMatcher(None, _norm_col(a), c_norm).ratio()))
            cand.sort(key=lambda x: x[1], reverse=True)
            if cand and cand[0][1] >= 0.72:
                best = cand[0][0]

        mapping[key] = best
    return mapping


def safe_to_numeric(s):
    # handle commas/spaces
    if isinstance(s, pd.Series):
        s2 = s.astype(str).str.replace(",", "", regex=False).str.strip()
        return pd.to_numeric(s2, errors="coerce")
    return pd.to_numeric(s, errors="coerce")


def safe_to_date(s):
    return pd.to_datetime(s, errors="coerce", infer_datetime_format=True)


def validate_mapping(mapping: dict):
    missing = []
    for key, required, label in SCHEMA:
        if required and (not mapping.get(key)):
            missing.append(f"{label} ({key})")
    return missing


def build_internal_df(raw_df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=raw_df.index)
    for key, _, _ in SCHEMA:
        col = mapping.get(key)
        out[key] = raw_df[col] if (col and col in raw_df.columns) else np.nan

    out["doc_date"] = safe_to_date(out["doc_date"]).dt.date.astype("object")

    for c in ["doc_id", "vendor", "city", "item", "category"]:
        out[c] = out[c].astype("string").fillna("").str.strip()

    return out


def compute_rules_and_scores(df, settings):
    for col in ["qty", "unit_price", "subtotal", "gst_rate", "gst_amount", "total", "payment_terms_days"]:
        if col in df.columns:
            df[col] = safe_to_numeric(df[col])

    # if gst_rate provided like 18 instead of 0.18
    df.loc[df["gst_rate"] > 1.0, "gst_rate"] = df.loc[df["gst_rate"] > 1.0, "gst_rate"] / 100.0

    df["expected_gst"] = (df["subtotal"] * df["gst_rate"]).round(2)
    df["expected_total"] = (df["subtotal"] + df["gst_amount"]).round(2)

    gst_tol = settings["gst_tolerance"]
    total_tol = settings["total_tolerance"]

    df["gst_mismatch_rule"] = (df["gst_amount"] - df["expected_gst"]).abs() > gst_tol
    df["total_mismatch_rule"] = (df["total"] - df["expected_total"]).abs() > total_tol

    dup_cols = ["doc_id", "vendor", "item", "qty", "unit_price", "subtotal", "gst_amount", "total"]
    dup_cols = [c for c in dup_cols if c in df.columns]
    df["duplicate_rule"] = df.duplicated(subset=dup_cols, keep="first") if dup_cols else False

    grp_key = ["vendor", "item"] if ("vendor" in df.columns and "item" in df.columns) else ["item"]
    df["unit_price_median"] = df.groupby(grp_key)["unit_price"].transform("median")
    df["unit_price_median"] = df["unit_price_median"].replace(0, np.nan)
    df["unit_price_dev"] = df["unit_price"] / df["unit_price_median"]

    over_pct = settings["overprice_pct"] / 100.0
    df["overprice_rule"] = (df["unit_price_dev"] > (1.0 + over_pct)) & df["unit_price_dev"].notna()

    rule_cols = ["gst_mismatch_rule", "total_mismatch_rule", "duplicate_rule", "overprice_rule"]
    df["pred_anomaly_rules"] = df[rule_cols].any(axis=1).astype(int)

    feats = pd.DataFrame(index=df.index)
    feats["qty"] = df["qty"].fillna(0)
    feats["unit_price_log"] = np.log1p(df["unit_price"].clip(lower=0).fillna(0))
    feats["subtotal_log"] = np.log1p(df["subtotal"].clip(lower=0).fillna(0))
    feats["total_log"] = np.log1p(df["total"].clip(lower=0).fillna(0))
    feats["gst_rate"] = df["gst_rate"].fillna(0)
    feats["terms"] = df.get("payment_terms_days", pd.Series(0, index=df.index)).fillna(0)
    feats["unit_price_dev"] = df["unit_price_dev"].fillna(1.0)

    contamination = settings["ml_contamination"]
    model = IsolationForest(n_estimators=250, contamination=contamination, random_state=42)

    if len(df) >= 20:
        model.fit(feats)
        normality = model.decision_function(feats)
        anomaly_strength = -normality
    else:
        anomaly_strength = np.zeros(len(df))

    a = np.asarray(anomaly_strength, dtype=float)
    if np.nanmax(a) - np.nanmin(a) > 1e-9:
        risk_score = (a - np.nanmin(a)) / (np.nanmax(a) - np.nanmin(a)) * 100.0
    else:
        risk_score = np.zeros_like(a)

    df["risk_score"] = np.round(risk_score, 1)

    risk_thr = settings["risk_threshold"]
    price_dev_thr = settings["price_dev_threshold"]
    qty_thr = settings["qty_high_percentile"]
    qty_cut = df["qty"].quantile(qty_thr) if df["qty"].notna().any() else np.inf

    df["needs_review"] = (
        (df["pred_anomaly_rules"] == 0)
        & (
            (df["risk_score"] >= risk_thr)
            | (df["unit_price_dev"] >= price_dev_thr)
            | (df["qty"] >= qty_cut)
        )
    ).astype(int)

    def _reason_row(r):
        if r["pred_anomaly_rules"] == 1:
            if r["gst_mismatch_rule"]:
                return "gst_mismatch_rule"
            if r["total_mismatch_rule"]:
                return "total_mismatch_rule"
            if r["duplicate_rule"]:
                return "duplicate_rule"
            if r["overprice_rule"]:
                return "overprice_rule"
            return "rule_flagged"
        if r["needs_review"] == 1:
            triggers = []
            if r["risk_score"] >= risk_thr:
                triggers.append("ml_risk")
            if pd.notna(r["unit_price_dev"]) and r["unit_price_dev"] >= price_dev_thr:
                triggers.append("price_dev")
            if pd.notna(r["qty"]) and r["qty"] >= qty_cut:
                triggers.append("qty_high")
            return "+".join(triggers) if triggers else "needs_review"
        return "none"

    df["final_reason"] = df.apply(_reason_row, axis=1)
    df["final_status"] = np.where(
        df["pred_anomaly_rules"] == 1, "FLAGGED",
        np.where(df["needs_review"] == 1, "NEEDS_REVIEW", "OK")
    )

    return df


# ---------------------------
# Header
# ---------------------------
logo_path = os.path.join("docs", "logo.png")

colA, colB = st.columns([0.75, 0.25], vertical_alignment="center")
with colA:
    st.markdown(
        """
        <div class="hero">
          <div>
            <span class="badge">🧠 Rules + ML</span>
            <span class="badge">🧾 CSV In → Scored CSV Out</span>
            <span class="badge">⚡ Fast audit</span>
          </div>
          <h1>Invoice / Quote Anomaly Detector</h1>
          <p>Upload a CSV → map columns → detect anomalies → export scored output</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with colB:
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
    else:
        st.caption("Tip: add `docs/logo.png` for a nicer header ✨")
    st.caption(f"Built by: {APP_OWNER} • Invoice/Quote Anomaly Detector")


# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.markdown("## ⚙️ Setup")
st.sidebar.caption(f"👤 Built by {APP_OWNER}")

uploaded = st.sidebar.file_uploader("📤 Upload CSV", type=["csv"])

sample_path = os.path.join("data", "sample_input.csv")
if os.path.exists(sample_path):
    with open(sample_path, "rb") as f:
        st.sidebar.download_button(
            "⬇️ Download sample CSV template",
            data=f,
            file_name="sample_input.csv",
            mime="text/csv",
        )

st.sidebar.markdown("---")


raw_df = None
if uploaded is not None:
    try:
        raw_df = pd.read_csv(uploaded)
    except Exception:
        raw_df = pd.read_csv(uploaded, encoding_errors="ignore")

mapping = {}
settings = {}
run_btn = False

if raw_df is not None and len(raw_df.columns) > 0:
    st.sidebar.markdown("## 🧩 Column mapping")

    if "mapping_state" not in st.session_state:
        st.session_state.mapping_state = suggest_mapping(list(raw_df.columns))

    if st.sidebar.button("✨ Auto-map columns"):
        st.session_state.mapping_state = suggest_mapping(list(raw_df.columns))

    cols_list = ["(not set)"] + list(raw_df.columns)

    for key, required, label in SCHEMA:
        default_col = st.session_state.mapping_state.get(key)
        default_idx = cols_list.index(default_col) if default_col in cols_list else 0
        chosen = st.sidebar.selectbox(
            f"{'✅' if required else '➕'} {label}",
            options=cols_list,
            index=default_idx,
            help=f"Map your CSV column to `{key}`",
            key=f"map_{key}",
        )
        mapping[key] = None if chosen == "(not set)" else chosen

    missing = validate_mapping(mapping)
    if missing:
        st.sidebar.error("Missing required mappings:\n- " + "\n- ".join(missing))

    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🎛️ Detection settings")

    settings["gst_tolerance"] = st.sidebar.number_input(
        "GST tolerance (PKR)",
        min_value=0.0, max_value=5000.0,
        value=1.0, step=0.5,
        help="Small rounding differences allowed",
    )
    settings["total_tolerance"] = st.sidebar.number_input(
        "Total tolerance (PKR)",
        min_value=0.0, max_value=5000.0,
        value=1.0, step=0.5,
        help="Small rounding differences allowed",
    )
    settings["overprice_pct"] = st.sidebar.slider(
        "Overprice threshold (%)",
        min_value=5, max_value=200, value=30, step=5,
        help="Flag if unit price is this much above baseline median",
    )
    settings["ml_contamination"] = st.sidebar.slider(
        "ML sensitivity (contamination)",
        min_value=0.01, max_value=0.20, value=0.06, step=0.01,
        help="Higher = more anomalies detected by ML",
    )
    settings["risk_threshold"] = st.sidebar.slider(
        "Risk score threshold (Needs Review)",
        min_value=0, max_value=100, value=80, step=1,
        help="Above this score → Needs Review (if not already Flagged by rules)",
    )
    settings["price_dev_threshold"] = st.sidebar.slider(
        "Price deviation threshold (Needs Review)",
        min_value=1.0, max_value=5.0, value=1.15, step=0.05,
        help="If unit_price / median >= this → Needs Review (if not rules-flagged)",
    )
    settings["qty_high_percentile"] = st.sidebar.slider(
        "Qty high percentile (Needs Review)",
        min_value=0.80, max_value=0.99, value=0.95, step=0.01,
        help="Qty above this percentile → Needs Review (if not rules-flagged)",
    )

    run_btn = st.sidebar.button("🚀 Run detection", type="primary")


# ---------------------------
# Main layout
# ---------------------------
left, right = st.columns([0.58, 0.42], vertical_alignment="top")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧾 Preview")
    if raw_df is None:
        st.info("Upload a CSV to start.")
    else:
        st.caption("Top rows from your uploaded file (before mapping).")
        st.dataframe(raw_df.head(15), use_container_width=True, height=350)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧭 How to use")
    st.markdown(
        """
        **1) Upload CSV** (client ka file)  
        **2) Map columns** (invoice# ho ya doc_id — sab map ho jata hai)  
        **3) Run detection**  
        **4) Filter results + download scored CSV**
        
        **Buckets:**
        - 🚩 **FLAGGED** = strong rule issue (GST/total/duplicate/overprice)
        - ⚠️ **NEEDS_REVIEW** = suspicious (ML/price/qty)
        - ✅ **OK** = looks normal
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------
# Run detection
# ---------------------------
if raw_df is not None and run_btn:
    missing = validate_mapping(mapping)
    if missing:
        st.error("Please fix column mapping in sidebar before running.")
    else:
        internal_df = build_internal_df(raw_df, mapping)
        scored = compute_rules_and_scores(internal_df, settings)

        # Add metadata columns (optional but useful)
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        scored["created_by"] = APP_OWNER
        scored["generated_at"] = generated_at

        base_cols = [k for k, _, _ in SCHEMA]
        extra_cols = [
            "expected_gst", "expected_total",
            "gst_mismatch_rule", "total_mismatch_rule", "duplicate_rule", "overprice_rule",
            "unit_price_median", "unit_price_dev",
            "pred_anomaly_rules", "risk_score", "needs_review",
            "final_reason", "final_status",
            "created_by", "generated_at",
        ]
        show_cols = [c for c in base_cols + extra_cols if c in scored.columns]

        flagged_n = int((scored["final_status"] == "FLAGGED").sum())
        review_n = int((scored["final_status"] == "NEEDS_REVIEW").sum())
        ok_n = int((scored["final_status"] == "OK").sum())

        st.markdown("---")
        st.subheader("📌 Results")

        m1, m2, m3 = st.columns(3)
        m1.metric("🚩 FLAGGED", flagged_n)
        m2.metric("⚠️ NEEDS_REVIEW", review_n)
        m3.metric("✅ OK", ok_n)

        filter_choice = st.selectbox("Filter", ["ALL", "FLAGGED", "NEEDS_REVIEW", "OK"], index=0)
        if filter_choice != "ALL":
            view_df = scored[scored["final_status"] == filter_choice].copy()
        else:
            view_df = scored.copy()

        st.dataframe(view_df[show_cols], use_container_width=True, height=420)

        # File names with your tag
        filtered_name = f"invoice_anomaly_scored_output__{APP_TAG}.csv"
        full_name = f"invoice_anomaly_scored_full__{APP_TAG}.csv"

        out_csv = view_df[show_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download scored CSV (filtered view)",
            data=out_csv,
            file_name=filtered_name,
            mime="text/csv",
        )

        full_csv = scored[show_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download FULL scored CSV",
            data=full_csv,
            file_name=full_name,
            mime="text/csv",
        )

        with st.expander("📖 What do these columns mean? (simple)"):
            st.markdown(
                """
                - **expected_gst**: subtotal × gst_rate  
                - **expected_total**: subtotal + gst_amount  
                - **gst_mismatch_rule**: GST mismatch (tolerance se bahar)  
                - **total_mismatch_rule**: Total mismatch (tolerance se bahar)  
                - **duplicate_rule**: duplicate line detected  
                - **overprice_rule**: price median baseline se zyada  
                - **unit_price_dev**: unit_price / median (1.00 = normal, 1.20 = 20% higher)  
                - **risk_score (0–100)**: ML suspiciousness score  
                - **needs_review**: suspicious (manual check)  
                - **final_status**: FLAGGED / NEEDS_REVIEW / OK  
                - **final_reason**: why it got flagged/reviewed  
                - **created_by / generated_at**: report metadata
                """
            )
else:
    if raw_df is not None:
        st.warning("Sidebar me mapping + settings check karo, phir **Run detection** dabao.")


st.markdown("---")
st.caption(f"Built by {APP_OWNER} • Streamlit • Rules + ML triage")
