import io
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from numpy.lib.stride_tricks import sliding_window_view

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="FnO Pattern Scanner",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .block-container {max-width: 760px; padding-top: 1.2rem; padding-bottom: 2rem;}
    div[data-testid="stForm"] {border: 1px solid rgba(128,128,128,.25); border-radius: 16px; padding: 16px;}
    div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
        width: 100%; min-height: 3.1rem; border-radius: 12px; font-weight: 700;
    }
    div[data-testid="stMetric"] {background: rgba(128,128,128,.08); padding: 12px; border-radius: 12px;}
    h1 {font-size: 1.8rem !important;}
</style>
""", unsafe_allow_html=True)

st.title("📈 Historical FnO Pattern Scanner")
st.caption("Search historical NSE F&O charts for patterns resembling the latest 60 completed candles.")

# Last-known fallback only; the app first attempts to read the current NSE F&O file.
FALLBACK_SYMBOLS = sorted(set("""
RELIANCE HDFCBANK ICICIBANK SBIN AXISBANK KOTAKBANK INDUSINDBK BAJFINANCE BAJAJFINSV
SBILIFE HDFCLIFE TCS INFY WIPRO HCLTECH LT ITC BHARTIARTL MARUTI TATAMOTORS M&M
TATASTEEL JSWSTEEL NTPC POWERGRID ONGC COALINDIA BPCL IOC GRASIM ULTRACEMCO
ADANIENT ADANIPORTS SUNPHARMA DRREDDY CIPLA DIVISLAB APOLLOHOSP ASIANPAINT TITAN
TRENT EICHERMOT HEROMOTOCO BAJAJ-AUTO TVSMOTOR HINDUNILVR BEL HAL BHEL PFC RECLTD
DLF SIEMENS ABB CUMMINSIND POLYCAB DIXON VOLTAS HAVELLS CHOLAFIN SHRIRAMFIN PNB
BANKBARODA CANBK UNIONBANK FEDERALBNK JINDALSTEL SAIL VEDL HINDALCO NMDC MOTHERSON
ASHOKLEY BOSCHLTD ZYDUSLIFE LUPIN AUROPHARMA TORNTPHARM PERSISTENT COFORGE LTIM
TECHM MPHASIS OFSS INDHOTEL IRCTC NAUKRI ZOMATO SBICARD
""".split()))

@st.cache_data(ttl=6*60*60, show_spinner=False)
def get_fno_symbols():
    url = "https://www.nseindia.com/static/products-services/equity-derivatives-contract-information"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,*/*",
        "Referer": "https://www.nseindia.com/",
    }
    try:
        session = requests.Session()
        session.headers.update(headers)
        page = session.get(url, timeout=20)
        page.raise_for_status()
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(page.text, "html.parser")
        links = [a.get("href", "") for a in soup.find_all("a", href=True)
                 if "fo_mktlots" in a.get("href", "").lower()]
        if not links:
            raise RuntimeError("NSE F&O file link not found")
        csv_url = links[0]
        if csv_url.startswith("/"):
            csv_url = "https://www.nseindia.com" + csv_url
        elif csv_url.startswith("//"):
            csv_url = "https:" + csv_url
        raw = session.get(csv_url, timeout=20)
        raw.raise_for_status()
        table = pd.read_csv(io.BytesIO(raw.content))
        table.columns = [str(c).strip().upper() for c in table.columns]
        sym_col = next(c for c in table.columns if "SYMBOL" in c)
        section = table.apply(
            lambda row: row.astype(str).str.contains("Individual Securities", case=False).any(),
            axis=1
        )
        if not section.any():
            raise RuntimeError("Individual Securities section not found")
        part = table.loc[section[section].index[0]:].iloc[1:]
        symbols = sorted({str(x).strip().upper() for x in part[sym_col].dropna()
                          if str(x).strip() and str(x).strip().upper() not in {"NAN", "SYMBOL"}})
        if len(symbols) < 50:
            raise RuntimeError("NSE file returned an unexpectedly small universe")
        return symbols, "Live NSE F&O list"
    except Exception:
        return FALLBACK_SYMBOLS, "Built-in fallback list (NSE live list temporarily unavailable)"

@st.cache_data(ttl=24*60*60, show_spinner=False)
def download_history(symbol):
    try:
        df = yf.download(
            symbol + ".NS", period="max", interval="1d",
            auto_adjust=True, progress=False, threads=False
        )
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        cols = ["Open", "High", "Low", "Close", "Volume"]
        df = df[cols].copy().dropna(subset=["Open", "High", "Low", "Close"])
        df.index = pd.to_datetime(df.index)
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
        return df[~df.index.duplicated(keep="last")].sort_index()
    except Exception:
        return pd.DataFrame()

def make_timeframe(df, tf):
    if tf == "1D":
        return df.copy()
    rule = "W-FRI" if tf == "1W" else "ME"
    return df.resample(rule).agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum"
    }).dropna(subset=["Open", "High", "Low", "Close"])

def clean_completed(df, tf):
    today = pd.Timestamp.now().normalize()
    if tf == "1D":
        return df.loc[df.index.normalize() < today]
    if tf == "1W":
        week_start = today - pd.Timedelta(days=today.weekday())
        return df.loc[df.index < week_start]
    return df.loc[(df.index.year < today.year) |
                  ((df.index.year == today.year) & (df.index.month < today.month))]

def dtw_distance(a, b, window=8):
    n, m = len(a), len(b)
    w = max(window, abs(n-m))
    prev = np.full(m+1, np.inf)
    prev[0] = 0.0
    for i in range(1, n+1):
        cur = np.full(m+1, np.inf)
        for j in range(max(1, i-w), min(m, i+w)+1):
            cur[j] = abs(a[i-1]-b[j-1]) + min(prev[j], cur[j-1], prev[j-1])
        prev = cur
    return float(prev[m] / (n+m))

def scan_patterns(reference_symbol, tf, symbols, progress):
    downloaded = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(download_history, s): s for s in symbols}
        for k, future in enumerate(as_completed(futures), 1):
            s = futures[future]
            raw = future.result()
            if not raw.empty:
                frame = make_timeframe(raw, tf)
                if len(frame) >= 61:
                    downloaded[s] = frame
            progress.progress(k / max(1, len(futures)), text=f"Loading historical data: {k}/{len(futures)} stocks")

    if reference_symbol not in downloaded:
        # If symbol is not in F&O universe, try its own data so error can be precise.
        raw = download_history(reference_symbol)
        if not raw.empty:
            downloaded[reference_symbol] = make_timeframe(raw, tf)
    if reference_symbol not in downloaded:
        raise ValueError(f"No historical price data found for {reference_symbol}.")

    target_df = clean_completed(downloaded[reference_symbol], tf)
    if len(target_df) < 60:
        raise ValueError(f"Only {len(target_df)} completed candles are available; 60 are required.")
    target = target_df.iloc[-60:][["Open", "High", "Low", "Close"]].to_numpy(float)
    if target[0, 3] <= 0:
        raise ValueError("Invalid reference price data.")
    target_norm = target / target[0, 3] - 1
    target_close = target_norm[:, 3]
    target_start = target_df.index[0] if len(target_df) == 60 else target_df.index[-60]

    candidate_rows = []
    total_windows = 0
    for symbol, df0 in downloaded.items():
        df = df0
        vals = df[["Open", "High", "Low", "Close"]].to_numpy(float)
        if len(vals) < 60 or not np.isfinite(vals).all():
            continue
        win = sliding_window_view(vals, window_shape=60, axis=0).transpose(0, 2, 1)
        ends = np.arange(59, len(vals))
        eligible = np.asarray(df.index[59:] < target_start)
        if not eligible.any():
            continue
        win, ends = win[eligible], ends[eligible]
        bases = win[:, 0, 3]
        valid = np.isfinite(win).all(axis=(1,2)) & (bases > 0)
        win, ends = win[valid], ends[valid]
        if not len(win):
            continue
        total_windows += len(win)
        norm = win / bases[:, None, None] - 1
        close = norm[:, :, 3]
        t = (target_close-target_close.mean())/(target_close.std()+1e-9)
        means = close.mean(axis=1, keepdims=True)
        stds = close.std(axis=1, keepdims=True)+1e-9
        structure = np.mean(np.abs((close-means)/stds - t[None,:]), axis=1)
        candle = np.mean(np.abs(norm-target_norm[None,:,:]), axis=(1,2))
        fast = .40*structure + .30*candle
        take = np.argsort(fast)[:min(100, len(fast))]
        for idx in take:
            end = int(ends[idx])
            start = end-59
            candidate_rows.append((symbol, start, end, float(structure[idx]), float(candle[idx]),
                                   df.index[start], df.index[end], norm[idx,:,3].copy()))

    if not candidate_rows:
        return pd.DataFrame(), downloaded, target, total_windows

    candidate_rows.sort(key=lambda x: .4*x[3]+.3*x[4])
    candidate_rows = candidate_rows[:100]
    results = []
    for symbol, start, end, structure, candle, pstart, pend, hist_close in candidate_rows:
        d = downloaded[symbol]
        dtw = dtw_distance(target_close, hist_close)
        combined = .40*structure + .30*candle + .30*dtw
        base_close = float(d["Close"].iloc[end])
        row = {
            "Stock": symbol,
            "Pattern start": pd.Timestamp(pstart).strftime("%Y-%m-%d"),
            "Pattern end": pd.Timestamp(pend).strftime("%Y-%m-%d"),
            "Similarity score": round(combined, 6),
            "Structure error": round(structure, 6),
            "Candle error": round(candle, 6),
            "DTW error": round(dtw, 6),
        }
        for h in [5, 10, 20, 40, 60]:
            pos = end + h
            row[f"After {h} candles (%)"] = (
                round((float(d["Close"].iloc[pos])/base_close-1)*100, 2)
                if pos < len(d) and base_close else np.nan
            )
        results.append(row)
    result = pd.DataFrame(results).sort_values("Similarity score").head(20).reset_index(drop=True)
    return result, downloaded, target, total_windows

symbols, universe_source = get_fno_symbols()
st.caption(f"Universe: {len(symbols)} stocks · {universe_source}")

with st.form("scanner_form"):
    symbol = st.text_input("Stock symbol", value="SBICARD", placeholder="Example: SBICARD").strip().upper()
    timeframe_label = st.selectbox("Timeframe", ["Daily (1D)", "Weekly (1W)", "Monthly (1M)"])
    submitted = st.form_submit_button("🔍 Find historical matches", use_container_width=True)

if submitted:
    if not re.fullmatch(r"[A-Z0-9&-]{1,25}", symbol):
        st.error("Enter a valid NSE symbol, for example SBICARD.")
    else:
        tf = {"Daily (1D)": "1D", "Weekly (1W)": "1W", "Monthly (1M)": "1M"}[timeframe_label]
        bar = st.progress(0, text="Preparing scanner…")
        try:
            with st.spinner("Searching historical NSE F&O charts. The first scan may take several minutes; later scans reuse cached data."):
                results, histories, target, total = scan_patterns(symbol, tf, symbols, bar)
            bar.empty()
            st.session_state["scan_results"] = results
            st.session_state["scan_histories"] = histories
            st.session_state["scan_target"] = target
            st.session_state["scan_symbol"] = symbol
            st.session_state["scan_tf"] = tf
            st.session_state["scan_total"] = total
        except Exception as e:
            bar.empty()
            st.error(str(e))

if "scan_results" in st.session_state:
    results = st.session_state["scan_results"]
    symbol = st.session_state["scan_symbol"]
    tf = st.session_state["scan_tf"]
    st.divider()
    st.subheader(f"Results: {symbol} · {tf}")
    st.caption(f"Compared against {st.session_state['scan_total']:,} eligible historical 60-candle windows. Lower similarity score means a closer mathematical match.")
    if results.empty:
        st.warning("No eligible historical matches were found for this pattern.")
    else:
        st.dataframe(results, use_container_width=True, hide_index=True)
        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download results (CSV)", csv, file_name=f"{symbol}_{tf}_historical_matches.csv", mime="text/csv", use_container_width=True)

        target = st.session_state["scan_target"]
        target_line = (target[:,3]/target[0,3]-1)*100
        histories = st.session_state["scan_histories"]
        st.subheader("Closest chart comparisons")
        for _, row in results.head(3).iterrows():
            s = row["Stock"]
            hist = histories[s]
            start_date = pd.Timestamp(row["Pattern start"])
            positions = np.where(hist.index.normalize() == start_date.normalize())[0]
            if not len(positions):
                continue
            start = int(positions[0])
            pattern = hist.iloc[start:start+60]
            if len(pattern) != 60:
                continue
            match_line = (pattern["Close"].to_numpy()/float(pattern["Close"].iloc[0])-1)*100
            future = hist.iloc[start+60:start+120]
            future_line = (future["Close"].to_numpy()/float(pattern["Close"].iloc[0])-1)*100
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(60)), y=target_line, mode="lines", name=f"{symbol} current pattern"))
            fig.add_trace(go.Scatter(x=list(range(60)), y=match_line, mode="lines", name=f"{s} historical match"))
            if len(future_line):
                fig.add_trace(go.Scatter(x=list(range(60, 60+len(future_line))), y=future_line, mode="lines", name="What happened next", line=dict(dash="dash")))
            fig.add_vline(x=59, line_dash="dot", line_color="gray")
            fig.update_layout(
                title=f"{s} · {row['Pattern start']} to {row['Pattern end']} · score {row['Similarity score']}",
                xaxis_title="Candles (60-candle pattern, then historical follow-through)",
                yaxis_title="Change from starting close (%)",
                height=340, margin=dict(l=10,r=10,t=55,b=35),
                legend=dict(orientation="h", y=-0.25)
            )
            st.plotly_chart(fig, use_container_width=True)
    st.info("Historical similarity is descriptive, not a prediction or trading recommendation. Future outcomes in the table are historical returns after old matches, not expected returns.")
else:
    st.markdown("#### How to use")
    st.markdown("1. Enter an NSE F&O stock symbol (for example `SBICARD`).\n2. Choose Daily, Weekly or Monthly.\n3. Tap **Find historical matches**.")
    st.caption("The scanner uses 60 completed candles, price-structure matching and DTW candle matching. Historical data availability varies by stock and data provider.")
