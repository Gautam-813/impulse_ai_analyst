"""
data_sync.py — Hugging Face Hub ↔ MT5 Delta Sync Module
Handles: load, gap check, delta fetch, merge, and push back to HF Hub.
"""

import os
import tempfile
import pytz
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone, timedelta

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
SYMBOLS    = ["XAUUSD", "EURUSD"]
TIMEFRAME  = "1m"

# MT5 symbol → Yahoo Finance ticker mapping
# Keeps data_sync.py self-contained (no import from app.py needed)
YAHOO_MAPPING = {
    "XAUUSD":    "GC=F",       # Gold Futures
    "EURUSD":    "EURUSD=X",   # Euro/USD FX
    "DXY":       "DX-Y.NYB",   # US Dollar Index
    "^NSEI":     "^NSEI",      # Nifty 50 (Identity)
    "^NSEBANK":  "^NSEBANK",   # Bank Nifty (Identity)
}

def _filename(symbol: str) -> str:
    return f"{symbol}_M1_Data.parquet"

# -----------------------------------------------------------------------
# 1. Load parquet from HF Hub
# -----------------------------------------------------------------------
def load_from_hf(repo_id: str, symbol: str, hf_token: str) -> pd.DataFrame:
    """Download parquet from Hugging Face Hub and return as DataFrame."""
    from huggingface_hub import hf_hub_download
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=_filename(symbol),
            repo_type="dataset",
            token=hf_token,
            force_download=False   # uses local cache if file unchanged
        )
        df = pd.read_parquet(local_path)
        return df
    except Exception as e:
        raise RuntimeError(f"[HF Load] Failed to load {symbol}: {e}")

# -----------------------------------------------------------------------
# 2. Get the last (most recent) timestamp in the dataframe
# -----------------------------------------------------------------------
def get_last_timestamp(df: pd.DataFrame) -> datetime:
    """Find the latest timestamp in the loaded parquet."""
    time_col = next((c for c in df.columns if c.lower() in ("time", "timestamp", "date")), None)
    if time_col is None:
        raise ValueError("No recognisable time column found in DataFrame.")

    series = pd.to_datetime(df[time_col], errors="coerce").dropna()
    if series.empty:
        raise ValueError("Time column contains no valid datetime values.")

    last = series.max()
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    return last

# -----------------------------------------------------------------------
# 3. Get data gap info (for display in sidebar)
# -----------------------------------------------------------------------
def get_gap_info(df: pd.DataFrame) -> dict:
    """Return a dict with last_timestamp, gap_hours, and a human label."""
    now = datetime.now(timezone.utc)
    last_ts = get_last_timestamp(df)
    gap_seconds = (now - last_ts).total_seconds()
    gap_hours   = gap_seconds / 3600
    gap_days    = gap_hours   / 24

    if gap_hours < 1:
        label = f"{int(gap_seconds / 60)} minutes old"
    elif gap_hours < 24:
        label = f"{gap_hours:.1f} hours old"
    else:
        label = f"{gap_days:.1f} days old"

    return {
        "last_timestamp": last_ts,
        "gap_hours":      round(gap_hours, 2),
        "gap_days":       round(gap_days, 2),
        "label":          label,
        "is_fresh":       gap_hours < 0.1   # < 6 minutes → no sync needed
    }

# -----------------------------------------------------------------------
# 4. Fetch ONLY the missing candles from the MT5 Data Server
# -----------------------------------------------------------------------
def fetch_delta_from_mt5(server_url: str, symbol: str,
                          from_time: datetime, to_time: datetime,
                          mt5_token: str = "") -> pd.DataFrame:
    """Call the local MT5 FastAPI server for gap candles only."""
    try:
        payload = {
            "symbol":     symbol,
            "timeframe":  TIMEFRAME,
            "start_date": from_time.isoformat(),
            "end_date":   to_time.isoformat()
        }
        headers = {"X-MT5-Token": mt5_token}
        resp = requests.post(f"{server_url.rstrip('/')}/data/fetch",
                             json=payload, headers=headers, timeout=90)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success"):
            raise RuntimeError(data.get("error", "MT5 server returned failure"))

        df = pd.DataFrame(data["data"])
        if df.empty:
            return df
        df["time"] = pd.to_datetime(df["time"])
        return df

    except requests.ConnectionError:
        raise RuntimeError(f"Cannot connect to MT5 server at {server_url}. "
                           f"Is mt5_data_server.py running?")
    except requests.Timeout:
        raise RuntimeError("MT5 server timed out. The gap may be too large — try a smaller range.")
    except Exception as e:
        raise RuntimeError(f"[MT5 Fetch] {e}")

# -----------------------------------------------------------------------
# 5. Merge new rows into existing DataFrame and push back to HF Hub
# -----------------------------------------------------------------------
def merge_and_push(existing_df: pd.DataFrame, delta_df: pd.DataFrame,
                   repo_id: str, symbol: str, hf_token: str) -> pd.DataFrame:
    """Merge, dedup, sort, and push the updated parquet to HF Hub."""
    from huggingface_hub import upload_file

    merged = pd.concat([existing_df, delta_df], ignore_index=True)

    time_col = next((c for c in merged.columns if c.lower() in ("time", "timestamp", "date")), "time")
    merged[time_col] = pd.to_datetime(merged[time_col], errors="coerce")
    merged = (merged
              .drop_duplicates(subset=[time_col])
              .sort_values(time_col)
              .reset_index(drop=True))

    # Write to a temp file, upload, then clean up
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".parquet")
    try:
        os.close(tmp_fd)
        merged.to_parquet(tmp_path, index=False)
        upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=_filename(symbol),
            repo_id=repo_id,
            repo_type="dataset",
            token=hf_token,
            commit_message=f"Delta sync {symbol} → {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return merged

# -----------------------------------------------------------------------
# 6. One-click full sync for a single symbol
# -----------------------------------------------------------------------
def sync_symbol(repo_id: str, symbol: str,
                hf_token: str, mt5_server_url: str,
                mt5_token: str = "") -> tuple[pd.DataFrame, dict]:
    """
    Full pipeline:
      load parquet → check gap → fetch delta → merge → push → return fresh df + stats
    """
    now = datetime.now(timezone.utc)

    # Step 1 — Load existing data
    existing_df = load_from_hf(repo_id, symbol, hf_token)

    # Step 2 — Check gap
    gap_info = get_gap_info(existing_df)
    if gap_info["is_fresh"]:
        return existing_df, {
            "status":    "already_fresh",
            "label":     gap_info["label"],
            "new_rows":  0,
        }

    # Step 3 — Fetch only missing delta from MT5
    from_time = gap_info["last_timestamp"]
    delta_df  = fetch_delta_from_mt5(mt5_server_url, symbol, from_time, now, mt5_token)

    if delta_df.empty:
        return existing_df, {
            "status":   "no_new_data",
            "label":    gap_info["label"],
            "new_rows": 0,
        }

    # Step 4 — Merge + Push
    updated_df = merge_and_push(existing_df, delta_df, repo_id, symbol, hf_token)

    return updated_df, {
        "status":   "synced",
        "label":    f"Up to date ({len(delta_df)} new rows added)",
        "new_rows": len(delta_df),
        "gap_hours": gap_info["gap_hours"],
    }

# -----------------------------------------------------------------------
# 7. Ping MT5 server health
# -----------------------------------------------------------------------
def ping_mt5_server(server_url: str, mt5_token: str = "") -> dict:
    """Returns status dict: {reachable, mt5_initialized, account, server}"""
    try:
        headers = {"X-MT5-Token": mt5_token}
        resp = requests.get(f"{server_url.rstrip('/')}/health", headers=headers, timeout=4)
        data = resp.json()
        return {
            "reachable":       True,
            "mt5_initialized": data.get("mt5_initialized", False),
        }
    except:
        return {"reachable": False, "mt5_initialized": False}

# -----------------------------------------------------------------------
# 8. Sync from Yahoo Finance (Global Markets)
# -----------------------------------------------------------------------
def get_gap_info(df):
    """
    Calculates the gap between the last candle in the dataframe and now.
    """
    import math
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {"gap_hours": 0, "label": "Empty Dataset", "is_fresh": False, "last_timestamp": None}
    
    try:
        time_col = next((c for c in df.columns if c.lower() in ("time", "timestamp", "date")), None)
        if not time_col: 
            return {"gap_hours": 0, "label": "Invalid Format", "is_fresh": False, "last_timestamp": None}

        # Last candle timestamp
        last_time = pd.to_datetime(df[time_col].iloc[-1], utc=True)
        if pd.isna(last_time):
            return {"gap_hours": 9999, "label": "NaT Error", "is_fresh": False, "last_timestamp": None}

        now = datetime.now(timezone.utc)
        diff = now - last_time
        gap_hours = diff.total_seconds() / 3600
        
        # Comprehensive check for NaN or Inf
        if not math.isfinite(gap_hours):
            return {"gap_hours": 9999, "label": "Time Error", "is_fresh": False, "last_timestamp": last_time}

        is_fresh = gap_hours < 0.25 # 15 min freshness
        
        # Build Label safely
        if gap_hours < 1: 
            label = "Up to date"
        elif gap_hours < 24: 
            label = f"{int(gap_hours)} hours ago"
        elif gap_hours < 8760: # up to 1 year
            label = f"{int(gap_hours/24)} days ago"
        else:
            label = "Old data"
            
        return {
            "gap_hours": float(gap_hours), 
            "label": label, 
            "last_timestamp": last_time, 
            "is_fresh": is_fresh
        }
    except Exception as e:
        print(f"DEBUG: Critical Gap check error: {e}")
        return {"gap_hours": 9999, "label": "Sync Error", "is_fresh": False, "last_timestamp": None}
def sync_yahoo_symbol(repo_id: str, symbol: str, hf_token: str,
                      existing_df: pd.DataFrame = None) -> tuple[pd.DataFrame, dict]:
    """
    Smart delta-sync from Yahoo Finance:
      1. Finds the last timestamp in existing_df (or HF Hub).
      2. Fetches ONLY the missing candles from that timestamp → now.
      3. Merges, deduplicates, sorts, and pushes the result back to HF Hub.

    Parameters
    ----------
    existing_df : pd.DataFrame, optional
        Already-loaded Hub DataFrame.  When supplied, the function skips the
        internal HF re-download entirely (no double-fetch).
    """
    from datetime import timezone
    from huggingface_hub import upload_file

    INTERVAL = "1m"          # M1 candles
    YF_MAX_DAYS_1M = 6       # Yahoo caps 1m history at ~7 days; use 6 to be safe

    def _yf_filename(s: str) -> str:
        """Reverse-map a Yahoo ticker back to the MT5-style parquet filename."""
        for mt5_sym, yh_sym in YAHOO_MAPPING.items():
            if s == yh_sym:
                return f"{mt5_sym}_M1_Data.parquet"
        return f"{s}_{INTERVAL}_Data.parquet"

    target_filename = _yf_filename(symbol)
    now_utc = datetime.now(timezone.utc)
    print(f"\n🔄 Smart delta-sync: {symbol}  →  {target_filename}")

    # ── Step A: Resolve the existing Hub data ──────────────────────────────
    if existing_df is not None and not existing_df.empty:
        hub_df = existing_df.copy()
        hub_df['time'] = pd.to_datetime(hub_df['time'], utc=True, errors='coerce')
        print(f"   📂 Using caller-supplied data. Rows on Hub: {len(hub_df):,}")
    else:
        try:
            from huggingface_hub import hf_hub_download
            local_cache = hf_hub_download(
                repo_id=repo_id, filename=target_filename,
                repo_type="dataset", token=hf_token, force_download=True
            )
            hub_df = pd.read_parquet(local_cache)
            hub_df['time'] = pd.to_datetime(hub_df['time'], utc=True)
            print(f"   ☁️  Downloaded Hub data. Rows: {len(hub_df):,}")
        except Exception as dl_err:
            print(f"   🆕 No existing Hub file ({target_filename}). Starting fresh. [{dl_err}]")
            hub_df = pd.DataFrame()

    # ── Step B: Determine exact start for the delta fetch ─────────────────
    if not hub_df.empty:
        last_ts = hub_df['time'].max()                  # last known candle
        gap_hours = (now_utc - last_ts).total_seconds() / 3600
        gap_label = (
            f"{int(gap_hours * 60)} min" if gap_hours < 1
            else f"{gap_hours:.1f} h" if gap_hours < 24
            else f"{gap_hours / 24:.1f} days"
        )
        print(f"   🕐 Last candle: {last_ts.strftime('%Y-%m-%d %H:%M UTC')}  |  Gap: {gap_label}")

        if gap_hours < 0.25:
            print("   ✅ Data is already fresh — no fetch needed.")
            return hub_df, {"status": "already_fresh", "new_rows": 0, "total_rows": len(hub_df),
                            "filename": target_filename}

        # Yahoo Finance 1m data: max look-back is ~7 days
        # Clamp start so we never request further back than Yahoo allows
        earliest_allowed = now_utc - pd.Timedelta(days=YF_MAX_DAYS_1M)
        fetch_start = pd.Timestamp(max(last_ts, earliest_allowed))
    else:
        # No existing data — fetch the max Yahoo allows for 1m
        fetch_start = pd.Timestamp(now_utc - pd.Timedelta(days=YF_MAX_DAYS_1M))
        gap_label = f"{YF_MAX_DAYS_1M} days (fresh start)"

    print(f"   📡 Fetching delta: {fetch_start.strftime('%Y-%m-%d %H:%M UTC')} → now")

    # ── Step C: Fetch ONLY the missing candles from Yahoo ─────────────────
    try:
        ticker = yf.Ticker(symbol)
        new_df = ticker.history(
            start=fetch_start.strftime("%Y-%m-%d"),   # yfinance needs date-only (YYYY-MM-DD)
            end=(now_utc + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),  # +1d so today is included
            interval=INTERVAL
        )
        if new_df.empty:
            print("   ⚠️  Yahoo returned 0 rows for the requested range.")
            return hub_df if not hub_df.empty else pd.DataFrame(), {
                "status": "no_new_data", "new_rows": 0,
                "total_rows": len(hub_df), "filename": target_filename
            }

        new_df = new_df.reset_index()
        time_col = next(
            (c for c in new_df.columns if c.lower() in ("datetime", "date")),
            new_df.columns[0]
        )
        new_df = new_df.rename(columns={
            time_col:       'time',
            'Open':         'open',
            'High':         'high',
            'Low':          'low',
            'Close':        'close',
            'Volume':       'tick_volume',
            'Dividends':    'dividends',
            'Stock Splits': 'splits',
        })
        new_df['time'] = pd.to_datetime(new_df['time'], utc=True)
        new_df.columns = [c.lower() for c in new_df.columns]
        print(f"   📥 Fetched {len(new_df):,} new candles from Yahoo Finance.")

    except Exception as e:
        raise RuntimeError(f"[Yahoo Fetch] {symbol}: {e}")

    # Step 2 — Merge with existing Hub data
    # Priority: use the caller-supplied existing_df first (avoids redundant HF download)
    if existing_df is not None and not existing_df.empty:
        print(f"   🔗 Using caller-supplied existing data ({len(existing_df):,} rows). Skipping HF re-download.")
        hub_df = existing_df.copy()
        hub_df['time'] = pd.to_datetime(hub_df['time'], utc=True, errors='coerce')
    else:
        # Fallback: download existing file from HF Hub
        try:
            from huggingface_hub import hf_hub_download
            local_cache = hf_hub_download(
                repo_id=repo_id,
                filename=target_filename,
                repo_type="dataset",
                token=hf_token,
                force_download=True   # always get latest
            )
            hub_df = pd.read_parquet(local_cache)
            hub_df['time'] = pd.to_datetime(hub_df['time'], utc=True)
            print(f"   ☁️  Downloaded existing Hub data ({len(hub_df):,} rows).")
        except Exception as dl_err:
            print(f"   🆕 No existing Hub file found ({target_filename}): {dl_err}. Starting fresh.")
            hub_df = pd.DataFrame()

    # ── Step D: Merge new delta into Hub data ─────────────────────────────
    if hub_df.empty:
        full_df = new_df
        print(f"   🆕 Starting fresh with {len(full_df):,} rows.")
    else:
        full_df = pd.concat([hub_df, new_df], ignore_index=True)
        full_df = (
            full_df
            .drop_duplicates(subset=['time'], keep='last')
            .sort_values('time')
            .reset_index(drop=True)
        )
        print(f"   ✨ Merged. Total rows: {len(full_df):,} (was {len(hub_df):,}, +{len(new_df):,} new candles)")

    # ── Step E: Push merged result back to HF Hub ──────────────────────────
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".parquet")
    try:
        os.close(tmp_fd)
        full_df.to_parquet(tmp_path, index=False)
        upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=target_filename,
            repo_id=repo_id,
            repo_type="dataset",
            token=hf_token,
            commit_message=f"Delta sync {symbol} ({INTERVAL}) → {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )
        print(f"   ☁️  Pushed {target_filename} to Hub successfully.")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return full_df, {
        "status": "synced",
        "new_rows": len(new_df),
        "total_rows": len(full_df),
        "filename": target_filename,
    }
# -----------------------------------------------------------------------
# 9. Direct Live Fetch (For Real-time Monitoring / Live Tab)
# -----------------------------------------------------------------------
def pull_mt5_window(server_url: str, symbol: str, timeframe: str = "1m", 
                    count: int = 500, token: str = "") -> pd.DataFrame:
    """
    Directly pulls the last X candles from MT5 for the Live Feed Tab.
    Does not touch Hugging Face. Pure broker source.
    """
    try:
        # We'll use the 'FetchData' endpoint with a relative range
        # Or even better, just use the 'today' or 'last_24h' preset
        now = datetime.now(timezone.utc)
        
        # Estimate how far back based on count and timeframe
        # TF minutes: 1m=1, 5m=5, etc.
        tf_mins = 1
        tf_str = str(timeframe)
        if tf_str.endswith('m'): tf_mins = int(tf_str[:-1]) if len(tf_str) > 1 else 1
        elif tf_str.endswith('h'): tf_mins = int(tf_str[:-1]) * 60 if len(tf_str) > 1 else 60
        elif tf_str.endswith('d'): tf_mins = 1440
        
        start_time = now - timedelta(minutes=(count * tf_mins) + 60) # +Buffer
        
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "start_date": start_time.isoformat(),
            "end_date": now.isoformat()
        }
        
        headers = {"X-MT5-Token": token}
        resp = requests.post(f"{server_url.rstrip('/')}/data/fetch", json=payload, headers=headers, timeout=10)
        
        if resp.status_code != 200:
            return pd.DataFrame()
            
        data = resp.json()
        if not data.get("success"):
            return pd.DataFrame()
            
        df = pd.DataFrame(data["data"])
        if df.empty:
            return df
            
        # Ensure 'time' is typed
        df['time'] = pd.to_datetime(df['time'], utc=True)
        return df.tail(count).reset_index(drop=True)
        
    except Exception as e:
        print(f"DEBUG: pull_mt5_window error: {e}")
        return pd.DataFrame()
