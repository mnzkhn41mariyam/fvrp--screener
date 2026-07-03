import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FRVP POC Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@400;500;600&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .main { background: #0d1117; color: #e6edf3; }
  .stApp { background: #0d1117; }

  .header-band {
    background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
    border-bottom: 1px solid #21262d;
    padding: 1.2rem 1.5rem 1rem;
    margin-bottom: 1.5rem;
  }
  .header-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
    color: #58a6ff;
    letter-spacing: -0.5px;
    margin: 0;
  }
  .header-sub {
    font-size: 0.78rem;
    color: #8b949e;
    margin-top: 3px;
    font-family: 'IBM Plex Mono', monospace;
  }

  .metric-row { display: flex; gap: 12px; margin-bottom: 1.2rem; flex-wrap: wrap; }
  .metric-box {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 0.8rem 1.1rem;
    min-width: 120px;
  }
  .metric-label { font-size: 0.68rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.8px; }
  .metric-value { font-family: 'IBM Plex Mono', monospace; font-size: 1.25rem; font-weight: 600; color: #e6edf3; margin-top: 2px; }
  .metric-value.green { color: #3fb950; }
  .metric-value.red { color: #f85149; }
  .metric-value.yellow { color: #d29922; }

  .signal-bull {
    background: #0f2a1a;
    border-left: 3px solid #3fb950;
    border-radius: 6px;
    padding: 0.6rem 0.9rem;
    margin-bottom: 0.5rem;
    font-size: 0.82rem;
  }
  .signal-bear {
    background: #2a0f0f;
    border-left: 3px solid #f85149;
    border-radius: 6px;
    padding: 0.6rem 0.9rem;
    margin-bottom: 0.5rem;
    font-size: 0.82rem;
  }
  .signal-sym { font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 0.88rem; }
  .signal-detail { color: #8b949e; font-size: 0.76rem; margin-top: 2px; }

  .upload-box {
    background: #161b22;
    border: 1px dashed #30363d;
    border-radius: 10px;
    padding: 2rem;
    text-align: center;
    color: #8b949e;
    font-size: 0.85rem;
  }

  .stDataFrame { background: #161b22 !important; }
  div[data-testid="stDataFrame"] { border: 1px solid #21262d; border-radius: 8px; }

  .info-tag {
    display: inline-block;
    background: #1c2a3a;
    color: #58a6ff;
    border-radius: 4px;
    padding: 1px 7px;
    font-size: 0.72rem;
    font-family: 'IBM Plex Mono', monospace;
    margin-right: 4px;
  }
  .warn-tag {
    display: inline-block;
    background: #2a1f0a;
    color: #d29922;
    border-radius: 4px;
    padding: 1px 7px;
    font-size: 0.72rem;
    font-family: 'IBM Plex Mono', monospace;
  }

  hr { border-color: #21262d; margin: 1rem 0; }
  h3 { color: #e6edf3 !important; font-size: 0.95rem !important; font-weight: 600 !important; }
  p, li { color: #8b949e; font-size: 0.83rem; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-band">
  <div class="header-title">📊 FRVP · POC Crossover Screener</div>
  <div class="header-sub">Fixed Range from 6-Month Pivot High · NSE EQ · Daily Bhavcopy</div>
</div>
""", unsafe_allow_html=True)


# ── Core logic ────────────────────────────────────────────────────────────────

def detect_pivot_high(highs: pd.Series, lookback: int = 5) -> int:
    """Return index of the last N-bar swing high within the series."""
    for i in range(len(highs) - lookback - 1, lookback - 1, -1):
        window = highs.iloc[i - lookback: i + lookback + 1]
        if highs.iloc[i] == window.max():
            return i
    # fallback: absolute high
    return highs.idxmax() if not highs.empty else 0


def compute_poc(df_range: pd.DataFrame, price_bins: int = 50) -> float:
    """
    Approximate POC from OHLCV data in the fixed range.
    Distributes volume across price bins; returns midpoint of highest-volume bin.
    """
    lo = df_range["LOW"].min()
    hi = df_range["HIGH"].max()
    if hi <= lo:
        return (hi + lo) / 2

    bins = np.linspace(lo, hi, price_bins + 1)
    vol_profile = np.zeros(price_bins)

    for _, row in df_range.iterrows():
        bar_lo, bar_hi, vol = row["LOW"], row["HIGH"], row["TOTTRDQTY"]
        bar_range = bar_hi - bar_lo
        if bar_range == 0:
            # All volume at close
            idx = np.searchsorted(bins, row["CLOSE_PRICE"]) - 1
            idx = min(max(idx, 0), price_bins - 1)
            vol_profile[idx] += vol
        else:
            # Distribute volume proportionally across covered bins
            for b in range(price_bins):
                overlap = min(bins[b + 1], bar_hi) - max(bins[b], bar_lo)
                if overlap > 0:
                    vol_profile[b] += vol * (overlap / bar_range)

    poc_bin = np.argmax(vol_profile)
    poc_price = (bins[poc_bin] + bins[poc_bin + 1]) / 2
    return poc_price


def screen_symbol(group: pd.DataFrame, pivot_lookback: int, price_bins: int):
    """
    For one symbol's history (sorted ascending by date):
    1. Find 6-month pivot high bar
    2. Compute POC from pivot high bar → latest bar
    3. Check if today's close crossed POC vs yesterday
    Returns dict with signal info or None.
    """
    group = group.sort_values("TIMESTAMP").reset_index(drop=True)
    if len(group) < 10:
        return None

    highs = group["HIGH"]
    pivot_idx = detect_pivot_high(highs, lookback=pivot_lookback)

    fixed_range = group.iloc[pivot_idx:]
    if len(fixed_range) < 3:
        return None

    poc = compute_poc(fixed_range, price_bins=price_bins)

    latest = group.iloc[-1]
    prev = group.iloc[-2]

    close_now = latest["CLOSE_PRICE"]
    close_prev = prev["CLOSE_PRICE"]

    bull_cross = (close_prev < poc) and (close_now >= poc)
    bear_cross = (close_prev > poc) and (close_now <= poc)

    if not (bull_cross or bear_cross):
        return None

    pivot_high_price = group.iloc[pivot_idx]["HIGH"]
    pivot_date = group.iloc[pivot_idx]["TIMESTAMP"]
    num_bars = len(fixed_range)

    return {
        "Symbol": latest["SYMBOL"],
        "Signal": "🟢 BULL" if bull_cross else "🔴 BEAR",
        "Close": round(close_now, 2),
        "POC": round(poc, 2),
        "POC Dist %": round((close_now - poc) / poc * 100, 2),
        "Pivot High": round(pivot_high_price, 2),
        "Pivot Date": str(pivot_date)[:10],
        "Range Bars": num_bars,
        "Volume": int(latest["TOTTRDQTY"]),
    }


def load_bhavcopy(file) -> pd.DataFrame:
    """Load and normalise a bhavcopy CSV. Supports both the legacy NSE
    bhavcopy format and the newer UDiFF 'Common Bhavcopy' format
    (columns like TCKRSYMB, TRADDT, TTLTRADGVOL etc.)."""
    content = file.read()
    try:
        text = content.decode("utf-8")
    except Exception:
        text = content.decode("latin-1")

    df = pd.read_csv(StringIO(text))
    df.columns = [c.strip().upper() for c in df.columns]

    is_new_format = "TCKRSYMB" in df.columns

    if is_new_format:
        # New UDiFF combined bhavcopy: filter to equity cash-market rows only,
        # since this file also bundles F&O / other instrument types.
        if "SGMT" in df.columns:
            df = df[df["SGMT"].astype(str).str.strip() == "CM"].copy()
        if "FININSTRMTP" in df.columns:
            df = df[df["FININSTRMTP"].astype(str).str.strip() == "STK"].copy()
        if "SCTYSRS" in df.columns:
            df = df[df["SCTYSRS"].astype(str).str.strip() == "EQ"].copy()

        rename_map = {
            "TCKRSYMB": "SYMBOL",
            "SCTYSRS": "SERIES",
            "TRADDT": "TIMESTAMP",
            "OPNPRIC": "OPEN",
            "HGHPRIC": "HIGH",
            "LWPRIC": "LOW",
            "CLSPRIC": "CLOSE_PRICE",
            "TTLTRADGVOL": "TOTTRDQTY",
        }
        df = df.rename(columns=rename_map)
    else:
        # Legacy bhavcopy format
        if "SERIES" in df.columns:
            df = df[df["SERIES"].str.strip() == "EQ"].copy()

        rename_map = {}
        for c in df.columns:
            if "CLOSE" in c:
                rename_map[c] = "CLOSE_PRICE"
            elif c in ("TOTTRDQTY", "VOLUME", "TTL_TRD_QNTY", "TOTAL_TRADED_QUANTITY"):
                rename_map[c] = "TOTTRDQTY"
            elif c == "DATE1" or c == "DATE":
                rename_map[c] = "TIMESTAMP"
        df = df.rename(columns=rename_map)

    for col in ["TIMESTAMP", "CLOSE_PRICE", "HIGH", "LOW", "OPEN", "TOTTRDQTY", "SYMBOL"]:
        if col not in df.columns:
            st.error(f"Column '{col}' not found. Columns present: {list(df.columns)}")
            return pd.DataFrame()

    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], dayfirst=True, errors="coerce")
    for col in ["CLOSE_PRICE", "HIGH", "LOW", "OPEN", "TOTTRDQTY"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["TIMESTAMP", "CLOSE_PRICE", "HIGH", "LOW", "TOTTRDQTY"])


# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    pivot_lookback = st.slider("Pivot High Lookback (bars)", 3, 15, 5,
                               help="N bars on each side to qualify a swing high")
    price_bins = st.slider("Volume Profile Bins", 20, 100, 50,
                           help="Price buckets for POC calculation (more = precise, slower)")
    min_volume = st.number_input("Min Volume Filter", value=50000, step=10000)
    signal_filter = st.radio("Signal Direction", ["Both", "Bullish Only", "Bearish Only"])
    st.markdown("---")
    st.markdown("<p>Upload multiple days' bhavcopy for accurate 6-month range.</p>", unsafe_allow_html=True)


# ── File upload ───────────────────────────────────────────────────────────────
st.markdown("### 📂 Upload Bhavcopy Files")
st.markdown("<p>Upload one or more daily bhavcopy CSV files (NSE EQ). More history = better POC accuracy.</p>",
            unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Drop bhavcopy CSV files here",
    type=["csv"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if not uploaded_files:
    st.markdown("""
    <div class="upload-box">
      📁 Drop NSE bhavcopy CSV files above<br>
      <span style="font-size:0.75rem">Supports standard NSE bhavcopy format · Multiple files for multi-day history</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📖 How it works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **1. Pivot High Detection**  
        Finds the last significant swing high in ~6 months using N-bar lookback logic.
        """)
    with col2:
        st.markdown("""
        **2. Volume Profile**  
        Distributes bar volume across price bins from pivot high → today. POC = highest volume bin.
        """)
    with col3:
        st.markdown("""
        **3. Crossover Signal**  
        Flags stocks where yesterday's close was below/above POC and today crossed it.
        """)
    st.stop()


# ── Process ───────────────────────────────────────────────────────────────────
with st.spinner("Loading & processing bhavcopy data…"):
    frames = []
    for f in uploaded_files:
        df_raw = load_bhavcopy(f)
        if not df_raw.empty:
            frames.append(df_raw)

    if not frames:
        st.error("Could not parse any uploaded file. Check column names.")
        st.stop()

    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["SYMBOL", "TIMESTAMP"])

    # Apply volume filter on latest day
    latest_date = df_all["TIMESTAMP"].max()
    latest_day = df_all[df_all["TIMESTAMP"] == latest_date]
    valid_syms = latest_day[latest_day["TOTTRDQTY"] >= min_volume]["SYMBOL"].unique()
    df_all = df_all[df_all["SYMBOL"].isin(valid_syms)]

    total_syms = df_all["SYMBOL"].nunique()
    date_range = f"{df_all['TIMESTAMP'].min().date()} → {df_all['TIMESTAMP'].max().date()}"

    results = []
    progress = st.progress(0)
    symbols = df_all["SYMBOL"].unique()

    for i, sym in enumerate(symbols):
        group = df_all[df_all["SYMBOL"] == sym]
        try:
            res = screen_symbol(group, pivot_lookback, price_bins)
            if res:
                results.append(res)
        except Exception:
            pass
        progress.progress((i + 1) / len(symbols))

    progress.empty()

# ── Results ───────────────────────────────────────────────────────────────────
if not results:
    st.warning("No POC crossover signals found for today. Try uploading more history or adjusting settings.")
    st.stop()

df_res = pd.DataFrame(results)

# Apply signal direction filter
if signal_filter == "Bullish Only":
    df_res = df_res[df_res["Signal"].str.contains("BULL")]
elif signal_filter == "Bearish Only":
    df_res = df_res[df_res["Signal"].str.contains("BEAR")]

bull_count = df_res["Signal"].str.contains("BULL").sum()
bear_count = df_res["Signal"].str.contains("BEAR").sum()

# ── Metrics ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="metric-row">
  <div class="metric-box">
    <div class="metric-label">Screened</div>
    <div class="metric-value">{total_syms}</div>
  </div>
  <div class="metric-box">
    <div class="metric-label">Signals</div>
    <div class="metric-value yellow">{len(df_res)}</div>
  </div>
  <div class="metric-box">
    <div class="metric-label">Bullish</div>
    <div class="metric-value green">{bull_count}</div>
  </div>
  <div class="metric-box">
    <div class="metric-label">Bearish</div>
    <div class="metric-value red">{bear_count}</div>
  </div>
  <div class="metric-box">
    <div class="metric-label">Date Range</div>
    <div class="metric-value" style="font-size:0.78rem">{date_range}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Signal cards ──────────────────────────────────────────────────────────────
st.markdown("### 🎯 Signals")

tab1, tab2, tab3 = st.tabs(["📋 Table", "🟢 Bullish", "🔴 Bearish"])

with tab1:
    df_display = df_res.sort_values("POC Dist %", key=abs)
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Close": st.column_config.NumberColumn(format="₹%.2f"),
            "POC": st.column_config.NumberColumn(format="₹%.2f"),
            "Pivot High": st.column_config.NumberColumn(format="₹%.2f"),
            "POC Dist %": st.column_config.NumberColumn(format="%.2f%%"),
            "Volume": st.column_config.NumberColumn(format="%d"),
        }
    )
    csv_out = df_display.to_csv(index=False)
    st.download_button("⬇️ Download Signals CSV", csv_out, "poc_signals.csv", "text/csv")

with tab2:
    bulls = df_res[df_res["Signal"].str.contains("BULL")].sort_values("POC Dist %")
    if bulls.empty:
        st.info("No bullish crossovers today.")
    for _, row in bulls.iterrows():
        st.markdown(f"""
        <div class="signal-bull">
          <span class="signal-sym">🟢 {row['Symbol']}</span>
          <div class="signal-detail">
            Close ₹{row['Close']} &nbsp;|&nbsp; POC ₹{row['POC']} &nbsp;|&nbsp;
            Dist {row['POC Dist %']:+.2f}% &nbsp;|&nbsp;
            Pivot {row['Pivot Date']} @ ₹{row['Pivot High']} &nbsp;|&nbsp;
            Range {row['Range Bars']} bars
          </div>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    bears = df_res[df_res["Signal"].str.contains("BEAR")].sort_values("POC Dist %", ascending=False)
    if bears.empty:
        st.info("No bearish crossovers today.")
    for _, row in bears.iterrows():
        st.markdown(f"""
        <div class="signal-bear">
          <span class="signal-sym">🔴 {row['Symbol']}</span>
          <div class="signal-detail">
            Close ₹{row['Close']} &nbsp;|&nbsp; POC ₹{row['POC']} &nbsp;|&nbsp;
            Dist {row['POC Dist %']:+.2f}% &nbsp;|&nbsp;
            Pivot {row['Pivot Date']} @ ₹{row['Pivot High']} &nbsp;|&nbsp;
            Range {row['Range Bars']} bars
          </div>
        </div>
        """, unsafe_allow_html=True)
    
