import os
from datetime import datetime, timedelta
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------
# Page / App config
# -----------------------------
st.set_page_config(page_title="Pocket Quant", page_icon="📊", layout="wide")

# -----------------------------
# Simple Authentication
# -----------------------------
# Use Streamlit Cloud secrets when available
# .streamlit/secrets.toml
# [auth]
# username = "your_username"
# password = "your_password"

DEFAULT_USER = "demo"
DEFAULT_PASS = "demo"

AUTH_USER = st.secrets.get("auth", {}).get("username", DEFAULT_USER)
AUTH_PASS = st.secrets.get("auth", {}).get("password", DEFAULT_PASS)

@st.cache_data(show_spinner=False)
def _auth_ok(u: str, p: str) -> bool:
    return (u == AUTH_USER) and (p == AUTH_PASS)

with st.sidebar:
    st.markdown("### 🔒 Giriş")
    u = st.text_input("Kullanıcı adı", value="", type="default")
    p = st.text_input("Şifre", value="", type="password")
    ok = st.button("Giriş")

if not ok or not _auth_ok(u, p):
    st.info("Lütfen sol taraftan giriş yapın. Demo için kullanıcı adı: demo, şifre: demo")
    st.stop()

# -----------------------------
# Helpers
# -----------------------------
NUMERIC_COLS_HINT = [
    "open", "high", "low", "close", "vwap", "volume", "spot_volume",
    "delta_volume", "cumulative_delta_volume", "sum_open_interest", "delta_oi",
    "cumulative_delta_oi", "taker_long_short_ratio", "funding_rate",
    "liquidation_position_volume", "liquidation_position_count", "big_trades",
    "trapped_oi", "trapped_volume", "trapped_vwap"
]

RENAME_MAP = {  # graceful fallbacks if columns are slightly different later
}

@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    # Normalize columns
    df.columns = [c.strip() for c in df.columns]
    if "timestamp" not in df.columns:
        raise ValueError("CSV'de 'timestamp' kolonu bulunamadı.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Coerce numerics for known numeric columns, keep categorical text (e.g., trapped_vwap_type)
    for col in df.columns:
        if col == "timestamp":
            continue
        if col in NUMERIC_COLS_HINT:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Add helpers
    df["date"] = df["timestamp"].dt.date
    return df

@st.cache_data(show_spinner=False)
def daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    # Daily OHLC from first/last and min/max; VWAP as volume-weighted; sums/means for others
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "vwap": lambda s: np.average(s, weights=df.loc[s.index, "volume"].fillna(0)) if s.notna().any() else np.nan,
        "volume": "sum",
        "spot_volume": "sum",
        "sum_open_interest": "last",
        "taker_long_short_ratio": "median",
        "funding_rate": "mean",
        "big_trades": "sum",
        "delta_volume": "mean",
        "delta_oi": "mean",
        "cumulative_delta_volume": "last",
        "cumulative_delta_oi": "last",
        "trapped_oi": "sum",
        "trapped_volume": "sum",
    }

    cols_for_agg = {k: v for k, v in agg.items() if k in df.columns}
    g = df.groupby(df["timestamp"].dt.date)
    out = g.agg(cols_for_agg)
    out.index = pd.to_datetime(out.index)
    out = out.sort_index().reset_index().rename(columns={"index": "day", "timestamp": "day"})
    out = out.rename(columns={"date": "day"})
    out["day"] = out["day"].dt.date
    return out

@st.cache_data(show_spinner=False)
def period_stats(df_period: pd.DataFrame) -> pd.DataFrame:
    # According to spec:
    # - price, vwap, volume, sum_oi and delta* (non-cumulative) -> averages
    # - cumulative* -> sums
    # - trapped* -> sums
    # - big_trades -> sum
    # - taker_long_short_ratio -> median

    out = {}
    # Identify columns dynamically
    cols = df_period.columns

    # Averages
    for c in ["close", "vwap", "volume", "sum_open_interest"]:
        if c in cols:
            out[f"avg_{c}"] = df_period[c].mean(skipna=True)

    # delta* averages (non-cumulative only)
    delta_cols = [c for c in cols if "delta" in c and "cumulative" not in c]
    for c in delta_cols:
        out[f"avg_{c}"] = df_period[c].mean(skipna=True)

    # cumulative* sums
    cumulative_cols = [c for c in cols if "cumulative" in c]
    for c in cumulative_cols:
        out[f"sum_{c}"] = df_period[c].sum(skipna=True)

    # trapped* sums
    trapped_cols = [c for c in cols if c.startswith("trapped_")]
    for c in trapped_cols:
        out[f"sum_{c}"] = df_period[c].sum(skipna=True)

    # big_trades total
    if "big_trades" in cols:
        out["sum_big_trades"] = df_period["big_trades"].sum(skipna=True)

    # taker_long_short_ratio median
    if "taker_long_short_ratio" in cols:
        out["median_taker_long_short_ratio"] = df_period["taker_long_short_ratio"].median(skipna=True)

    return pd.DataFrame([out])

@st.cache_data(show_spinner=False)
def first_4h_mask(day_df: pd.DataFrame) -> pd.Series:
    # first 4 hours from the first timestamp of the day
    if day_df.empty:
        return pd.Series([], dtype=bool)
    t0 = day_df["timestamp"].iloc[0]
    return (day_df["timestamp"] < (t0 + pd.Timedelta(hours=4)))

@st.cache_data(show_spinner=False)
def prepare_day_table(day_df: pd.DataFrame) -> pd.DataFrame:
    # Compute 4h averages for some metrics, then build the requested table in order
    mask4 = first_4h_mask(day_df)

    def avg4(col):
        if col in day_df.columns and mask4.any():
            return day_df.loc[mask4, col].mean(skipna=True)
        return np.nan

    avg_volume_4h = avg4("volume")
    avg_delta_volume_4h = avg4("delta_volume")
    avg_delta_oi_4h = avg4("delta_oi")
    avg_big_trades_4h = avg4("big_trades")

    # vwap scaled to match first 5-min close, then follow its own relative changes
    if "vwap" in day_df.columns and "close" in day_df.columns:
        if not day_df["vwap"].isna().all() and not day_df["close"].isna().all():
            v0 = day_df["vwap"].iloc[0]
            c0 = day_df["close"].iloc[0]
            scale = c0 / v0 if (pd.notna(v0) and v0 != 0) else 1.0
            day_df = day_df.copy()
            day_df["vwap_scaled"] = day_df["vwap"] * scale
        else:
            day_df = day_df.copy()
            day_df["vwap_scaled"] = np.nan
    else:
        day_df = day_df.copy()
        day_df["vwap_scaled"] = np.nan

    # close - vwap diff
    if "close" in day_df.columns and "vwap" in day_df.columns:
        day_df["close_minus_vwap"] = day_df["close"] - day_df["vwap"]
    else:
        day_df["close_minus_vwap"] = np.nan

    # Build ordered view
    wanted = [
        "close", "vwap", "close_minus_vwap",
        "volume", "spot_volume", "avg_volume_4h",
        "delta_volume", "avg_delta_volume_4h", "cumulative_delta_volume",
        "sum_open_interest", "delta_oi", "avg_delta_oi_4h", "cumulative_delta_oi",
        "trapped_oi", "trapped_volume", "trapped_vwap", "trapped_vwap_type",
        "trapped_oi_sequence", "big_trades", "avg_big_trades_4h", "taker_long_short_ratio",
    ]

    # Fill constant avg columns
    day_df["avg_volume_4h"] = avg_volume_4h
    day_df["avg_delta_volume_4h"] = avg_delta_volume_4h
    day_df["avg_delta_oi_4h"] = avg_delta_oi_4h
    day_df["avg_big_trades_4h"] = avg_big_trades_4h

    # Only keep columns that exist; if a requested column is missing, create NaN placeholder
    for c in wanted:
        if c not in day_df.columns:
            day_df[c] = np.nan

    out = day_df[["timestamp"] + wanted].copy()
    return out

# -----------------------------
# UI
# -----------------------------
st.title("📊 Pocket Quant – VWAP & OI Mini")
st.caption("CSV içeriğinden tam dönem grafik, dönem özeti ve gün içi 5dk detayları")

with st.sidebar:
    st.markdown("### 📥 CSV Yükle")
    up = st.file_uploader("CSV dosyanızı seçin", type=["csv"], accept_multiple_files=False)

if not up:
    st.warning("Devam etmek için CSV yükleyin.")
    st.stop()

# Load
try:
    df = load_csv(up.getvalue())
except Exception as e:
    st.error(f"CSV yüklenemedi: {e}")
    st.stop()

# -----------------------------
# Home: Full-period price graph + period picker
# -----------------------------
st.subheader("1) Tüm Dönem – Fiyat Grafiği")
cols = st.columns([4, 1])
with cols[0]:
    fig_full = go.Figure()
    if "close" in df.columns:
        fig_full.add_trace(go.Scatter(x=df["timestamp"], y=df["close"], name="Close", mode="lines"))
    if "vwap" in df.columns:
        fig_full.add_trace(go.Scatter(x=df["timestamp"], y=df["vwap"], name="VWAP", mode="lines"))
    fig_full.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_full, use_container_width=True)

with cols[1]:
    min_day = df["timestamp"].min().date()
    max_day = df["timestamp"].max().date()
    st.markdown("**Dönem Seçimi**")
    start = st.date_input("Başlangıç", value=min_day, min_value=min_day, max_value=max_day, key="start")
    end = st.date_input("Bitiş", value=max_day, min_value=min_day, max_value=max_day, key="end")

if start > end:
    st.error("Başlangıç tarihi bitişten büyük olamaz.")
    st.stop()

mask_period = (df["timestamp"].dt.date >= start) & (df["timestamp"].dt.date <= end)
df_period = df.loc[mask_period].copy()

st.divider()
st.subheader("2) Dönem Özeti (Grafik + İstatistikler)")

# Period graph (Close + VWAP)
fig_period = go.Figure()
if not df_period.empty:
    if "close" in df_period.columns:
        fig_period.add_trace(go.Scatter(x=df_period["timestamp"], y=df_period["close"], name="Close", mode="lines"))
    if "vwap" in df_period.columns:
        fig_period.add_trace(go.Scatter(x=df_period["timestamp"], y=df_period["vwap"], name="VWAP", mode="lines"))
fig_period.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig_period, use_container_width=True)

# Period stats table
ps = period_stats(df_period)
st.markdown("**Dönem İstatistikleri**")
st.dataframe(ps, use_container_width=True)

# Day picker within period
st.markdown("**Gün seç**")
valid_days = sorted(pd.unique(df_period["timestamp"].dt.date))
if not valid_days:
    st.info("Seçilen dönemde veri yok.")
    st.stop()
sel_day = st.selectbox("Gün", options=valid_days, index=len(valid_days)-1)

# -----------------------------
# Day view: 5-min chart (vwap scaled) + detailed table
# -----------------------------
df_day = df_period[df_period["timestamp"].dt.date == sel_day].copy()

st.divider()
st.subheader(f"3) {sel_day} – 5 Dakikalık Grafik (VWAP fiyata endeksli)")

# Prepare scaled VWAP for the day
if not df_day.empty:
    v0 = df_day["vwap"].iloc[0] if "vwap" in df_day.columns else np.nan
    c0 = df_day["close"].iloc[0] if "close" in df_day.columns else np.nan
    scale = (c0 / v0) if (pd.notna(v0) and v0 not in (0, np.inf, -np.inf)) else 1.0

    fig_day = go.Figure()
    if "close" in df_day.columns:
        fig_day.add_trace(go.Scatter(x=df_day["timestamp"], y=df_day["close"], mode="lines", name="Close"))
    if "vwap" in df_day.columns:
        fig_day.add_trace(go.Scatter(x=df_day["timestamp"], y=df_day["vwap"] * scale, mode="lines", name="VWAP (scaled)"))
    if "volume" in df_day.columns:
        # Add volume on secondary y
        fig_day.add_trace(go.Bar(x=df_day["timestamp"], y=df_day["volume"], name="Volume", opacity=0.3, yaxis="y2"))
        fig_day.update_layout(
            yaxis=dict(title="Price"),
            yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False)
        )
    fig_day.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_day, use_container_width=True)
else:
    st.info("Seçilen gün için veri bulunamadı.")

# Detailed table with requested columns and 4h averages
st.subheader("4) Gün İçi 5dk Tablosu (istenen kolon sırası)")
day_table = prepare_day_table(df_day)
st.dataframe(day_table, use_container_width=True, height=420)

# Downloads
st.download_button(
    label="⬇️ Dönem İstatistiklerini İndir (CSV)",
    data=ps.to_csv(index=False).encode("utf-8"),
    file_name=f"period_stats_{start}_to_{end}.csv",
    mime="text/csv",
)

st.download_button(
    label=f"⬇️ {sel_day} Gün İçi Tabloyu İndir (CSV)",
    data=day_table.to_csv(index=False).encode("utf-8"),
    file_name=f"intraday_{sel_day}.csv",
    mime="text/csv",
)
