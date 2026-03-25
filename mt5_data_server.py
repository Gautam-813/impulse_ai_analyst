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
import os
import signal
import sys
import io
import socket
import uvicorn

# Optional Quant libraries for advanced analytic endpoints
try:
    import mplfinance as mpf
except ImportError:
    pass

# ============================================================================
# Dynamic Server Metadata
# ============================================================================
def get_network_ip():
    """Detect the primary network IP of this machine."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't need real connectivity, just triggers OS to resolve local interface
        s.connect(('8.8.8.8', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

SERVER_IP = get_network_ip()
def get_startup_config():
    # Priority 1: Environment Variables (Prevents blocking for EXEs/headless)
    port_env = os.getenv("MT5_SERVER_PORT")
    token_env = os.getenv("MT5_API_TOKEN")
    terminal_env = os.getenv("MT5_TERMINAL_PATH")

    if port_env and token_env:
        return int(port_env), token_env, terminal_env
    
    # Priority 2: Interactive CLI (Fallback if no environment is set)
    print("\n" + "=" * 60)
    print("      🚀 MT5 Data Server v2.6 — Configuration Startup")
    print("=" * 60 + "\n")

    default_port = os.getenv("MT5_SERVER_PORT", "5000")
    port_input = input(f"🔹 Enter Port [Default {default_port}]: ") or default_port
    port = int(port_input)

    default_token = os.getenv("MT5_API_TOKEN", "impulse_secure_2026")
    token_input = input(f"🔹 Enter Security Token [Default '{default_token}']: ") or default_token
    
    print("\n🔹 Multiple MT5 Instances Detected?")
    print("   (Leave empty to use your default/active MT5)")
    terminal_path = input("🔹 Enter MT5 Terminal Path (e.g. C:\\...\\terminal64.exe): ").strip() or None

    return port, token_input, terminal_path

PORT, MT5_API_TOKEN, STARTUP_PATH = get_startup_config()

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

    def resolve_symbol_name(self, target_name):
        """
        Smart resolution: XAUUSD -> XAUUSDm, XAUUSD.sc, etc.
        """
        # 1. Direct match
        if mt5.symbol_info(target_name):
            return target_name

        # 2. Case-insensitive Regex search
        try:
            all_symbols = mt5.symbols_get()
            if all_symbols:
                # Try regex match for things like XAUUSDm, XAUUSD.pro
                regex = re.compile(re.escape(target_name), re.IGNORECASE)
                for s in all_symbols:
                    if regex.search(s.name):
                        print(f"🔍 Auto-Resolved '{target_name}' → '{s.name}'")
                        return s.name
                
                # Special case for GOLD -> XAUUSD
                if target_name.upper() == "GOLD":
                    regex_xau = re.compile("XAUUSD", re.IGNORECASE)
                    for s in all_symbols:
                        if regex_xau.search(s.name):
                            print(f"🔍 Auto-Resolved 'GOLD' → '{s.name}'")
                            return s.name
                
                # Special case for XAUUSD -> GOLD
                if target_name.upper() == "XAUUSD":
                    regex_gold = re.compile("GOLD", re.IGNORECASE)
                    for s in all_symbols:
                        if regex_gold.search(s.name):
                            print(f"🔍 Auto-Resolved 'XAUUSD' → '{s.name}'")
                            return s.name
        except:
            pass
        
        return target_name # fallback

    def fetch_ohlc_data(self, symbol, timeframe, start_date, end_date):
        """
        Fetch OHLC data for given parameters
        """
        if not self.initialized:
            return None, "MT5 not initialized"
        
        try:
            # Smart Resolve Symbol Name
            symbol = self.resolve_symbol_name(symbol)

            # Enable symbol if not visible
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
    # Initial attempt to connect MT5 with startup path
    print("\n" + "=" * 60)
    print("  MT5 Data Server — Interactive + Secure Mode")
    print("=" * 60)
    
    if provider.initialize_mt5(STARTUP_PATH):
        print(f"✅ Auto-Connected to MT5 Terminal")
        acc = mt5.account_info()
        if acc: print(f"   Account: {acc.login} | Server: {acc.server}")
    else:
        print(f"⚠️  Manual MT5 initialization required (Call /initialize via API)")

    print(f"\n🚀 API Server starting on http://{SERVER_IP}:{PORT}")
    print(f"🔑 Security Token ACTIVE: {MT5_API_TOKEN}")
    print(f"📜 Docs (Swagger UI): http://{SERVER_IP}:{PORT}/docs")
    print("=" * 60 + "\n")

    # Clean Port Disposal Handler
    def signal_handler(sig, frame):
        print("\n🛑 Shutting down server and releasing port...")
        provider.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)

    uvicorn.run(app, host="0.0.0.0", port=PORT)
