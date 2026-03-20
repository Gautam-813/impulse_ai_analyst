"""
data_sync.py — Hugging Face Hub ↔ MT5 Delta Sync Module
Handles: load, gap check, delta fetch, merge, and push back to HF Hub.
"""

import os
import tempfile
import pytz
import requests
import pandas as pd
from datetime import datetime, timezone

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
SYMBOLS    = ["XAUUSD", "EURUSD"]
TIMEFRAME  = "1m"

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
