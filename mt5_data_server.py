"""
MT5 Data Server - On-Demand OHLC Data Provider
Provides data directly as JSON (no file storage)
Powered by FastAPI + Uvicorn
"""
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import pytz
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Annotated
import re
import uvicorn
import os

# ============================================================================
# Security Configuration
# ============================================================================
MT5_API_TOKEN = os.getenv("MT5_API_TOKEN", "impulse_secure_2026")

async def verify_token(x_mt5_token: Annotated[str | None, Header()] = None):
    if x_mt5_token != MT5_API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing MT5 Security Token")
    return x_mt5_token

# ============================================================================
# FastAPI App
# ============================================================================
app = FastAPI(
    title="MT5 Data Server",
    description="On-Demand OHLC Data Provider for MetaTrader 5",
    version="2.0.0"
)

# Allow all origins so Streamlit (on any host) can call this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Pydantic Request Models (replaces Flask request.json)
# ============================================================================
class InitializeRequest(BaseModel):
    terminal_path: Optional[str] = None

class SymbolSearchRequest(BaseModel):
    pattern: str

class FetchDataRequest(BaseModel):
    symbol: str
    timeframe: str   # 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M
    start_date: str  # ISO format: 2026-03-19T00:00:00Z
    end_date: str

class QuickFetchRequest(BaseModel):
    symbol: str
    preset: str = "yesterday_to_now"  # yesterday_to_now, last_24h, last_week, today, last_hour
    timeframe: str = "1m"

# ============================================================================
# MT5 Provider Class (unchanged from original)
# ============================================================================
class MT5DataProvider:
    def __init__(self):
        self.initialized = False
        self.terminal_path = None

    def initialize_mt5(self, terminal_path=None):
        """Initialize MT5 connection"""
        if self.initialized:
            return True
        try:
            if terminal_path:
                if not mt5.initialize(path=terminal_path):
                    return False
            else:
                if not mt5.initialize():
                    return False
            self.initialized = True
            self.terminal_path = terminal_path
            return True
        except Exception as e:
            print(f"MT5 initialization error: {e}")
            return False

    def shutdown(self):
        """Shutdown MT5 connection"""
        if self.initialized:
            mt5.shutdown()
            self.initialized = False

    def find_symbols(self, pattern):
        """Find symbols matching pattern (e.g., 'XAUUSD' finds XAUUSD, XAUUSDm, XAUUSD.sc)"""
        if not self.initialized:
            return []
        try:
            all_symbols = mt5.symbols_get()
            if all_symbols is None:
                return []
            regex_pattern = re.compile(pattern, re.IGNORECASE)
            matched = []
            for symbol in all_symbols:
                if regex_pattern.search(symbol.name):
                    matched.append({
                        'name': symbol.name,
                        'description': symbol.description,
                        'path': symbol.path,
                        'visible': symbol.visible
                    })
            return matched
        except Exception as e:
            print(f"Symbol search error: {e}")
            return []

    def get_symbol_info(self, symbol):
        """Get detailed symbol information"""
        if not self.initialized:
            return None
        try:
            info = mt5.symbol_info(symbol)
            if info is None:
                return None
            return {
                'name': info.name,
                'description': info.description,
                'point': info.point,
                'digits': info.digits,
                'trade_contract_size': info.trade_contract_size,
                'visible': info.visible
            }
        except Exception as e:
            print(f"Symbol info error: {e}")
            return None

    def fetch_ohlc_data(self, symbol, timeframe, start_date, end_date):
        """Fetch OHLC data for given parameters"""
        if not self.initialized:
            return None, "MT5 not initialized"
        try:
            if not mt5.symbol_select(symbol, True):
                return None, f"Failed to select symbol: {symbol}"

            timeframe_map = {
                '1m':  mt5.TIMEFRAME_M1,
                '5m':  mt5.TIMEFRAME_M5,
                '15m': mt5.TIMEFRAME_M15,
                '30m': mt5.TIMEFRAME_M30,
                '1h':  mt5.TIMEFRAME_H1,
                '4h':  mt5.TIMEFRAME_H4,
                '1d':  mt5.TIMEFRAME_D1,
                '1w':  mt5.TIMEFRAME_W1,
                '1M':  mt5.TIMEFRAME_MN1
            }
            tf = timeframe_map.get(timeframe, mt5.TIMEFRAME_M1)

            timezone = pytz.timezone("Etc/UTC")
            if isinstance(start_date, str):
                start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            else:
                start_dt = start_date
            if isinstance(end_date, str):
                end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            else:
                end_dt = end_date

            if start_dt.tzinfo is None:
                start_dt = timezone.localize(start_dt)
            if end_dt.tzinfo is None:
                end_dt = timezone.localize(end_dt)

            rates = mt5.copy_rates_range(symbol, tf, start_dt, end_dt)
            if rates is None or len(rates) == 0:
                return None, f"No data available for {symbol}"

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df, None
        except Exception as e:
            return None, f"Data fetch error: {str(e)}"

# Global provider instance
provider = MT5DataProvider()

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/health")
def health_check(token: Annotated[str, Depends(verify_token)]):
    """Check if server is running and MT5 status"""
    return {
        "status": "running",
        "mt5_initialized": provider.initialized,
        "terminal_path": provider.terminal_path
    }

@app.post("/initialize")
def initialize(req: InitializeRequest, token: Annotated[str, Depends(verify_token)]):
    """Initialize MT5 connection"""
    success = provider.initialize_mt5(req.terminal_path)
    if success:
        account_info = mt5.account_info()
        return {
            "success": True,
            "message": "MT5 initialized successfully",
            "account": account_info.login if account_info else None,
            "server": account_info.server if account_info else None
        }
    raise HTTPException(status_code=500, detail={
        "success": False,
        "message": "Failed to initialize MT5",
        "error": str(mt5.last_error())
    })

@app.post("/symbols/search")
def search_symbols(req: SymbolSearchRequest, token: Annotated[str, Depends(verify_token)]):
    """Search for symbols matching a regex pattern"""
    if not provider.initialized:
        raise HTTPException(status_code=400, detail="MT5 not initialized")
    if not req.pattern:
        raise HTTPException(status_code=400, detail="Pattern required")
    symbols = provider.find_symbols(req.pattern)
    return {
        "pattern": req.pattern,
        "count": len(symbols),
        "symbols": symbols
    }

@app.get("/symbols/info/{symbol}")
def symbol_info(symbol: str, token: Annotated[str, Depends(verify_token)]):
    """Get detailed symbol information"""
    if not provider.initialized:
        raise HTTPException(status_code=400, detail="MT5 not initialized")
    info = provider.get_symbol_info(symbol)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")
    return info

@app.post("/data/fetch")
def fetch_data(req: FetchDataRequest, token: Annotated[str, Depends(verify_token)]):
    """
    Fetch OHLC data and return directly as JSON.

    Timeframes: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M
    Dates: ISO format e.g. 2026-03-19T00:00:00Z
    """
    if not provider.initialized:
        raise HTTPException(status_code=400, detail="MT5 not initialized. Call /initialize first")

    df, error = provider.fetch_ohlc_data(
        symbol=req.symbol,
        timeframe=req.timeframe,
        start_date=req.start_date,
        end_date=req.end_date
    )
    if error:
        raise HTTPException(status_code=500, detail=error)

    df['time'] = df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    data_records = df.to_dict('records')

    print("\n" + "="*70)
    print(f"✓ Fetched {len(data_records)} candles for {req.symbol}")
    print(f"  Timeframe: {req.timeframe}")
    print(f"  Range: {df['time'].iloc[0]} → {df['time'].iloc[-1]}")
    print("="*70 + "\n")

    return {
        "success": True,
        "symbol": req.symbol,
        "timeframe": req.timeframe,
        "rows": len(data_records),
        "columns": list(df.columns),
        "date_range": {
            "start": df['time'].iloc[0],
            "end": df['time'].iloc[-1]
        },
        "data": data_records
    }

@app.post("/data/quick-fetch")
def quick_fetch(req: QuickFetchRequest, token: Annotated[str, Depends(verify_token)]):
    """
    Quick fetch with common presets.

    Presets: yesterday_to_now, last_24h, last_week, today, last_hour
    """
    if not provider.initialized:
        raise HTTPException(status_code=400, detail="MT5 not initialized")

    now = datetime.now(pytz.UTC)
    presets = {
        'yesterday_to_now': (now - timedelta(days=1), now),
        'last_24h':         (now - timedelta(hours=24), now),
        'last_week':        (now - timedelta(days=7), now),
        'today':            (now.replace(hour=0, minute=0, second=0, microsecond=0), now),
        'last_hour':        (now - timedelta(hours=1), now)
    }

    if req.preset not in presets:
        raise HTTPException(status_code=400, detail=f"Invalid preset. Choose from: {list(presets.keys())}")

    start_date, end_date = presets[req.preset]
    df, error = provider.fetch_ohlc_data(req.symbol, req.timeframe, start_date, end_date)
    if error:
        raise HTTPException(status_code=500, detail=error)

    df['time'] = df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    data_records = df.to_dict('records')

    print("\n" + "="*70)
    print(f"✓ Quick Fetch: {req.preset}")
    print(f"  Symbol: {req.symbol} | TF: {req.timeframe} | Candles: {len(data_records)}")
    print(f"  Range: {df['time'].iloc[0]} → {df['time'].iloc[-1]}")
    print("="*70 + "\n")

    return {
        "success": True,
        "symbol": req.symbol,
        "timeframe": req.timeframe,
        "preset": req.preset,
        "rows": len(data_records),
        "columns": list(df.columns),
        "date_range": {
            "start": df['time'].iloc[0],
            "end": df['time'].iloc[-1]
        },
        "data": data_records
    }

@app.post("/shutdown")
def shutdown_mt5(token: Annotated[str, Depends(verify_token)]):
    """Shutdown MT5 connection"""
    provider.shutdown()
    return {"success": True, "message": "MT5 connection closed"}

# ============================================================================
# Entry Point
# ============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("  MT5 Data Server v2.0 — FastAPI + Uvicorn")
    print("=" * 70)
    print("\n  Swagger UI:   http://localhost:5000/docs")
    print("  ReDoc UI:     http://localhost:5000/redoc")
    print("\n  Endpoints:")
    print("    POST  /initialize        — Connect to MT5 terminal")
    print("    POST  /symbols/search    — Search symbols by pattern")
    print("    GET   /symbols/info/{s}  — Get symbol details")
    print("    POST  /data/fetch        — Fetch OHLC (custom date range)")
    print("    POST  /data/quick-fetch  — Fetch OHLC (quick preset)")
    print("    GET   /health            — Server health check")
    print("    POST  /shutdown          — Disconnect from MT5")
    print("\n" + "=" * 70)

    uvicorn.run("mt5_data_server:app", host="0.0.0.0", port=5000, reload=True)
