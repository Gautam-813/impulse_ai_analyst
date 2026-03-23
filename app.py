
import streamlit as st
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import json
import io
import sys
import contextlib
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import re
import math
import scipy
import altair as alt
import sklearn
from openai import OpenAI
import pandas_ta as ta
import statsmodels.api as sm
import quantstats as qs
import duckdb
import polars as pl
import xgboost as xgb
from streamlit_autorefresh import st_autorefresh

# Lazy/Optional Heavy Imports
try:
    import pygwalker as pyg
    PYG_AVAILABLE = True
except (ImportError, MemoryError):
    PYG_AVAILABLE = False

try:
    from st_aggrid import AgGrid
    AGGRID_AVAILABLE = True
except (ImportError, MemoryError):
    AGGRID_AVAILABLE = False

# Symbol Mapping (MT5 -> Yahoo Finance)
YAHOO_MAPPING = {
    "XAUUSD": "GC=F",      # Gold Futures
    "EURUSD": "EURUSD=X",  # Euro/USD FX
    "DXY": "DX-Y.NYB",    # US Dollar Index
    "^NSEI": "^NSEI",     # Nifty 50 (Identity)
    "^NSEBANK": "^NSEBANK" # Bank Nifty (Identity)
}

# Page configuration
# ── PRO APP BRANDING ──
st.set_page_config(
    page_title="AI Quant Analyst | Impulse Master",
    page_icon="💰",
    layout="wide",
)

# --- 🚀 CLOUD DETECTION ---
import os
IS_CLOUD = os.environ.get("STREAMLIT_SERVER_PORT") is not None or "HOSTNAME" in os.environ
# -------------------------
# App theme and styling
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
        color: #ccd6f6;
    }
    .stChatMessage {
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Password Protection
def check_password():
    """Check if the user has entered the correct password."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    def login():
        entered_password = st.session_state.password_input
        if entered_password == st.secrets.get("PASSWORD", ""):
            st.session_state.authenticated = True
        else:
            st.session_state.login_error = True
    
    def logout():
        st.session_state.authenticated = False
        st.session_state.login_error = False
        st.session_state.password_input = ""
        st.rerun()
    
    if not st.session_state.authenticated:
        st.markdown("""
        <style>
        .login-container {
            max-width: 400px;
            margin: 100px auto;
            padding: 40px;
            border-radius: 10px;
            background: #161b22;
            border: 1px solid #30363d;
        }
        .login-title {
            text-align: center;
            color: #ccd6f6;
            margin-bottom: 30px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.markdown('<div class="login-container">', unsafe_allow_html=True)
            st.markdown('<h2 class="login-title">🔐 Login Required</h2>', unsafe_allow_html=True)
            st.text_input(
                "Password",
                type="password",
                key="password_input",
                on_change=login,
                placeholder="Enter access password"
            )
            if st.session_state.get("login_error", False):
                st.error("❌ Incorrect password. Please try again.")
            st.markdown('</div>', unsafe_allow_html=True)
        return False
    
    # Show logout button and EA download for authenticated users
    with st.sidebar:
        st.markdown("---")
        # EA Files Download
        st.subheader("📦 Download EA Files")
        st.caption("Download the MA Impulse Logger EA for generating datasets")
        
        import zipfile
        import os
        
        if st.button("📥 Download EA Package (ZIP)", width="stretch"):
            # Create ZIP file in memory
            zip_buffer = io.BytesIO()
            base_dir = os.path.dirname(os.path.abspath(__file__))
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add .ex5 file
                ex5_path = os.path.join(base_dir, "MA_Impulse_Logger.ex5")
                if os.path.exists(ex5_path):
                    zip_file.write(ex5_path, "MA_Impulse_Logger.ex5")
                
                # Add .mq5 file
                mq5_path = os.path.join(base_dir, "MA_Impulse_Logger.mq5")
                if os.path.exists(mq5_path):
                    zip_file.write(mq5_path, "MA_Impulse_Logger.mq5")
            
            zip_buffer.seek(0)
            st.download_button(
                label="✅ Click to Download",
                data=zip_buffer,
                file_name="MA_Impulse_Logger_Package.zip",
                mime="application/zip",
                key="download_ea_zip"
            )
        
        st.markdown("---")
        if st.button("🚪 Logout", width="stretch"):
            logout()
    
    return True

# Check authentication before showing app
if not check_password():
    st.stop()

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    provider_choice = st.selectbox("Select AI Provider", [
        "NVIDIA", "Groq", "OpenRouter", "OpenRouter (Free)", "Gemini", "GitHub Models", "Cerebras", "Bytez"
    ])
    
    # Leave value empty to prevent dots being shown in the DOM (security measure)
    input_api_key = st.text_input(f"{provider_choice} API Key", value="", type="password", help=f"Enter your {provider_choice} API Key.")
    
    # Resolve the key: User input takes priority, secret is used as a silent fallback
    secret_key_name = f"{provider_choice.upper().replace(' ', '_').replace('(', '').replace(')', '')}_API_KEY"
    if provider_choice == "OpenRouter" or provider_choice == "OpenRouter (Free)":
        secret_key_name = "OPEN_ROUTER_API_KEY"
    elif provider_choice == "GitHub Models":
        secret_key_name = "GITHUB_API_KEY"
    
    api_key_to_use = input_api_key if input_api_key else st.secrets.get(secret_key_name, "")
    
    if provider_choice == "NVIDIA":
        model_choice = st.selectbox("Select Model", [
            "qwen/qwen3.5-122b-a10b",
            "qwen/qwen2.5-coder-32b-instruct",
            "deepseek-ai/deepseek-v3.1",
            "deepseek-ai/deepseek-r1-distill-qwen-32b",
            "nvidia/llama-3.1-405b-instruct"
        ])
        base_url = "https://integrate.api.nvidia.com/v1"
    elif provider_choice == "Groq":
        model_choice = st.selectbox("Select Model", [
            "llama3-70b-8192",
            "llama3-8b-8192",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
            "deepseek-r1-distill-llama-70b"
        ])
        base_url = "https://api.groq.com/openai/v1"
    elif provider_choice == "OpenRouter":
        model_choice = st.selectbox("Select Model", [
            "anthropic/claude-3.5-sonnet",
            "anthropic/claude-3-5-haiku",
            "openrouter/hunter-alpha",
            "meta-llama/llama-3.1-70b-instruct",
            "google/gemini-pro-1.5",
            "mistralai/mixtral-8x22b-instruct"
        ])
        base_url = "https://openrouter.ai/api/v1"
    elif provider_choice == "OpenRouter (Free)":
        model_choice = st.selectbox("Select Free Model", [
            "deepseek/deepseek-r1:free",
            "openrouter/hunter-alpha",
            "google/gemini-pro-1.5:free",
            "meta-llama/llama-3.1-8b-instruct:free",
            "mistralai/mistral-nemo:free"
        ])
        base_url = "https://openrouter.ai/api/v1"
    elif provider_choice == "GitHub Models":
        model_choice = st.selectbox("Select Free Model", [
            "gpt-4o",
            "gpt-4o-mini",
            "meta-llama-3.1-70b-instruct",
            "AI21-Jamba-1.5-Large"
        ])
        base_url = "https://models.inference.ai.azure.com"
    elif provider_choice == "Cerebras":
        model_choice = st.selectbox("Select Free Model", [
            "llama3.1-70b",
            "llama3.1-8b"
        ])
        base_url = "https://api.cerebras.ai/v1"
    elif provider_choice == "Gemini":
        model_choice = st.selectbox("Select Model", [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-1.5-flash",
            "gemini-1.5-pro"
        ])
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    elif provider_choice == "Bytez":
        model_choice = st.selectbox("Select Model", [
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "Qwen/Qwen1.5-72B-Chat"
        ])
        base_url = "https://api.bytez.com/models/v2/openai/"

    enable_thinking = st.checkbox("Enable Thinking Mode", value=True)
    show_debug = st.checkbox("Show AI Trace (Debug)", value=False)
    
    st.markdown("---")
    st.subheader("🕵️ AI Connection Diagnostic")
    if st.button("Run Connection Test"):
        if not api_key_to_use:
            st.error("No API Key found! Please enter one.")
        else:
            try:
                test_key = api_key_to_use
                if provider_choice == "NVIDIA" and not test_key.startswith("nvapi-"):
                    test_key = f"nvapi-{test_key}"
                
                test_client = OpenAI(base_url=base_url, api_key=test_key)
                test_client.models.list()
                st.success("✅ Connection Successful!")
            except Exception as e:
                st.error(f"❌ Connection Failed: {e}")
    
    st.markdown("---")
    st.caption("🔒 **Security Note**: Your Entered API Key is processed in volatile memory only and is never saved to disk or logs.")
    st.markdown("---")
    st.subheader("🛠️ Pro Stack Status")
    if PYG_AVAILABLE: st.success("✅ Visual Explorer Ready")
    else: st.warning("⚠️ Visual Explorer Disabled (Memory)")
    
    if AGGRID_AVAILABLE: st.success("✅ Data Grid Ready")
    else: st.warning("⚠️ Data Grid Disabled (Memory)")
    
    st.markdown("---")
    st.subheader("🧠 Memory & Context")
    if st.button("🗑️ Clear AI Memory Context", width="stretch"):
        st.session_state.messages = []
        st.rerun()

    # ── Live Data Sync Panel ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📡 Live MT5 Sync")
    
    # Initialize lock state
    if "mt5_enabled" not in st.session_state:
        st.session_state.mt5_enabled = False

    if not st.session_state.mt5_enabled:
        st.info("MT5 Engine is currently Locked 🔒")
        if st.button("🔓 Activate MT5 Connection", width="stretch"):
            st.session_state.mt5_enabled = True
            st.rerun()
    else:
        if st.button("🔒 Lock MT5 Engine", width="stretch"):
            st.session_state.mt5_enabled = False
            st.rerun()
        
        st.caption("Connect your local MT5 server to pull only the missing candles.")

        mt5_url = st.text_input("MT5 Server URL",
                                 value=st.session_state.get("mt5_url", "http://localhost:5000"),
                                 key="mt5_url_input",
                                 placeholder="http://localhost:5000")

        mt5_token = st.text_input("MT5 Security Token",
                                   value=st.secrets.get("MT5_API_TOKEN", "impulse_secure_2026"),
                                   type="password")
        
        if st.button("🔌 Test Connection", width="stretch"):
            from data_sync import ping_mt5_server
            result = ping_mt5_server(mt5_url, mt5_token)
            if result["reachable"] and result["mt5_initialized"]:
                st.success("✅ MT5 Connected")
                st.session_state.mt5_url = mt5_url
            else:
                st.error("❌ MT5 Offline")

    st.markdown("---")
    data_source_priority = st.radio(
        "🌐 **Default Data Source**",
        ["☁️ Cloud Hub (Hugging Face)", "💾 Local Disk (.parquet)"],
        index=0,
        help="Choose where the 'Load' button looks first."
    )
    st.session_state.data_source_priority = data_source_priority

    if st.button("🔌 Test MT5 Connection", width="stretch"):
        from data_sync import ping_mt5_server
        result = ping_mt5_server(mt5_url, mt5_token)
        if result["reachable"] and result["mt5_initialized"]:
            st.success("✅ MT5 Server Connected & Ready")
            st.session_state.mt5_url = mt5_url
        elif result["reachable"]:
            st.warning("⚠️ Server running but MT5 not initialized yet")
        else:
            st.error("❌ Cannot reach server. Is mt5_data_server.py running?")

    st.markdown("---")

    # Per-symbol sync buttons
    hf_repo  = st.secrets.get("HF_REPO_ID", "")
    hf_token = st.secrets.get("HuggingFace_API_KEY", "")

    for sym in ["XAUUSD", "EURUSD"]:
        col_label, col_btn = st.columns([3, 2])
        with col_label:
            # Show data freshness if parquet is loaded
            try:
                if f"df_{sym}" in st.session_state:
                    from data_sync import get_gap_info
                    gap = get_gap_info(st.session_state[f"df_{sym}"])
                    if gap.get("gap_hours", 0) < 1:
                        st.caption(f"🟢 {sym} — Fresh")
                    elif gap.get("gap_hours", 0) < 24:
                        st.caption(f"🟡 {sym} — {gap['label']}")
                    else:
                        st.caption(f"🔴 {sym} — {gap['label']}")
                else:
                    st.caption(f"📊 {sym} — not loaded")
            except Exception as loop_err:
                st.caption(f"📊 {sym} (Status error)")
                # Don't crash the sidebar even if status check fails

        with col_btn:
            try:
                # ONLY SHOW SYNC BUTTON IF MT5 IS ACTIVATED
                if st.session_state.get("mt5_enabled", False):
                    if st.button(f"🔄 Sync", key=f"sync_{sym}", width="stretch"):
                        if not hf_repo or not hf_token:
                            st.error("Credential Error")
                        elif not mt5_url:
                            st.error("Enter MT5 setup")
                        else:
                            from data_sync import sync_symbol
                            with st.spinner(f"Syncing…"):
                                updated_df, stats = sync_symbol(hf_repo, sym, hf_token, mt5_url, mt5_token)
                                st.session_state[f"df_{sym}"] = updated_df
                                st.success(f"✅ {sym} ok")
                else:
                    st.button(f"🔄 Sync (Locked)", key=f"sync_locked_{sym}", width="stretch", disabled=True)
            except Exception as sync_err:
                st.error("Sync Errored")

    # --- ⚡ AUTO-SYNC CONTROLLER ---
    st.markdown("---")
    st.subheader("⚡ Auto-Sync Mode")
    auto_sync_on = st.toggle("Enable Auto-Sync 📡", value=st.session_state.get("auto_sync_on", False))
    st.session_state.auto_sync_on = auto_sync_on

    if auto_sync_on:
        from streamlit_autorefresh import st_autorefresh
        sync_interval = st.selectbox("Frequency ⏱️", [1, 5, 15, 30, 60], index=2, format_func=lambda x: f"Every {x} min")
        refresh_count = st_autorefresh(interval=sync_interval * 60 * 1000, key="sync_counter")
        
        if refresh_count > 0:
            current_sym = st.session_state.get("current_symbol_view", "XAUUSD")
            if hf_repo and hf_token and mt5_url:
                from data_sync import sync_symbol
                try:
                    updated_df, stats = sync_symbol(hf_repo, current_sym, hf_token, mt5_url, mt5_token)
                    st.session_state[f"df_{current_sym}"] = updated_df
                    st.toast(f"🔄 Auto-Synced {current_sym} ({stats.get('new_rows', 0)} new candles)")
                except Exception as e:
                    st.toast(f"🚨 Auto-Sync failed for {current_sym}")

    # --- 🌍 GLOBAL MARKET VAULT ---
    st.markdown("---")
    st.subheader("🌍 Global Market Vault")
    st.caption("Pull stocks/crypto from Yahoo Finance and archive to Cloud Hub.")

    yf_symbol = st.text_input("Ticker Symbol", placeholder="e.g. NVDA, BTC-USD, TSLA")
    
    col_per, col_int = st.columns(2)
    with col_per:
        yf_period = st.selectbox("History", ["1mo", "3mo", "1y", "5y", "max"], index=4)
    with col_int:
        yf_interval = st.selectbox("Interval", ["1h", "1d", "1wk"], index=1)

    if st.button("📥 Fetch & Archive to Cloud", width="stretch"):
        if not yf_symbol:
            st.warning("Please enter a ticker symbol.")
        elif not hf_repo or not hf_token:
            st.error("Missing Hugging Face credentials in secrets.toml")
        else:
            from data_sync import sync_yahoo_symbol
            with st.spinner(f"Fetching {yf_symbol} from Global Markets..."):
                try:
                    df_yf, stats_yf = sync_yahoo_symbol(hf_repo, yf_symbol, hf_token, yf_period, yf_interval)
                    st.success(f"✅ {yf_symbol} Vaulted: {stats_yf['total_rows']:,} rows in Cloud")
                    st.session_state.df = df_yf
                    st.session_state.file_name = stats_yf["filename"]
                    st.toast(f"🏆 {yf_symbol} added to your Cloud Warehouse!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Global Fetch Failed: {e}")

    # --- 🌐 GLOBAL WEB INTEL SEARCH ---
    st.markdown("---")
    st.subheader("🌐 Global Web Intel")
    st.caption("AI-powered research for macro news, sentiment, and geopolitics.")

    search_query = st.text_input("Research Topic", placeholder="e.g. FOMC meeting interest rates, XAUUSD sentiment")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        max_res = st.slider("Max Results", 3, 10, 5)
    with col_s2:
        search_type = st.radio("Focus", ["Market News", "General Search"], index=0, horizontal=True)

    if st.button("🔍 Deep Intel Search", width="stretch"):
        if not search_query:
            st.warning("Please enter a research topic.")
        else:
            from web_search import run_web_search, get_market_news, analyze_sentiment
            with st.spinner(f"🔍 AI is researching the web for '{search_query}'..."):
                try:
                    if search_type == "Market News":
                        res = get_market_news(search_query, max_results=max_res)
                    else:
                        res = run_web_search(search_query, max_results=max_res)
                    
                    if not res:
                        st.warning("No results found. Try a different query.")
                    else:
                        sentiment = analyze_sentiment(res)
                        st.info(f"🧠 **AI Sentiment Estimate:** {sentiment}")
                        
                        st.markdown("### 📰 Latest Findings:")
                        for r in res:
                            with st.expander(f"📌 {r['title']}"):
                                st.write(f"**Source:** {r['href']}")
                                st.write(f"{r['body']}")
                                st.markdown(f"[Read full article]({r['href']})")
                        st.toast("✅ Web Research Completed!")
                except Exception as e:
                    st.error(f"❌ Search Error: {e}")

# Arrow-safe display helper (fixes PyArrow serialization errors)
def make_arrow_safe(df):
    """Convert DataFrame to Arrow-compatible types for Streamlit display."""
    if not isinstance(df, pd.DataFrame):
        return df
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object or str(out[col].dtype).startswith('datetime64'):
            # Convert object and datetime to string (Arrow-safe)
            out[col] = out[col].astype(str)
        elif 'mixed' in str(out[col].dtype).lower() or 'period' in str(out[col].dtype).lower():
            out[col] = out[col].astype(str)
    return out

# Data Processing logic
def process_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.parquet'):
            # Parquet files are strongly typed and natively supported
            df = pd.read_parquet(uploaded_file)
        else:
            # Load EVERYTHING as string first to prevent 'object' type confusion
            df = pd.read_csv(uploaded_file, dtype=str, low_memory=False)
        
        # Manually convert known numeric columns (support both Segmented and MA Impulse formats)
        numeric_cols = [
            'cross_price', 'segment_index', 'segment_price', 
            'segment_size_price', 'segment_move_points', 
            'segment_move_percent', 'sequence_extreme_price', 'is_final',
            'DifferencePoints', 'DifferencePercent', 'MovingAveragePeriod',
            'CrossoverStartPrice', 'CrossoverEndPrice', 'AbsolutePeakPrice',
            'ImpulsePeakPrice', 'ReversalPrice', 'MA_At_AbsolutePeak',
            'MA_At_ImpulsePeak', 'MA_At_Reversal', 'DeepestRetracePrice',
            'MA_At_DeepestRetrace'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ensure 'symbol' and 'session' are clean strings
        for col in ['symbol', 'session', 'cross_type', 'segment_direction']:
            if col in df.columns:
                df[col] = df[col].fillna("UNKNOWN").astype(str)
        
        # Convert timestamps (naive datetime64[ns] for Arrow compatibility)
        time_cols = [
            'cross_time', 'cross_end_time', 'segment_time', 'sequence_extreme_time',
            'CrossoverStartTime', 'CrossoverEndTime', 'AbsolutePeakTime',
            'ImpulsePeakTime', 'ReversalTime', 'DeepestRetraceTime'
        ]
        for col in time_cols:
            if col in df.columns:
                dt = pd.to_datetime(df[col], errors='coerce', utc=False)
                df[col] = dt
                if df[col].dtype == object:
                    df[col] = df[col].astype(str)

        # FINAL ARROW COMPATIBILITY CHECK: 
        # Convert any remaining 'object' columns to string
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str)
        
        return df.copy()
    except Exception as e:
        st.error(f"☢️ Nuclear Data Clean Error: {e}")
        return None

# Streamlit wrapper that auto-sanitizes DataFrames for Arrow compatibility
class _SafeStreamlit:
    def __getattr__(self, name):
        return getattr(st, name)
    def write(self, *args, **kwargs):
        sanitized = [make_arrow_safe(a) if isinstance(a, pd.DataFrame) else a for a in args]
        return st.write(*sanitized, **kwargs)
    def dataframe(self, data, **kwargs):
        data = make_arrow_safe(data) if isinstance(data, pd.DataFrame) else data
        return st.dataframe(data, **kwargs)
    def table(self, data, **kwargs):
        data = make_arrow_safe(data) if isinstance(data, pd.DataFrame) else data
        return st.table(data, **kwargs)

# Code Execution Engine
def execute_generated_code(code, df):
    safe_st = _SafeStreamlit()
    # Prepare environment
    env = {
        'pd': pd, 
        'np': np,
        'px': px, 
        'go': go, 
        'plt': plt, 
        'sns': sns,
        'st': safe_st, 
        'df': df,
        'math': math,
        'scipy': scipy,
        'sklearn': sklearn,
        'pyg': pyg if PYG_AVAILABLE else None,
        'ta': ta,
        'sm': sm,
        'qs': qs,
        'duck': duckdb,
        'pl': pl,
        'alt': alt,
        'xgb': xgb
    }
    output = io.StringIO()
    # Close any existing plots
    plt.close('all')
    with contextlib.redirect_stdout(output):
        try:
            exec(code, env)
            return output.getvalue(), None
        except Exception as e:
            return output.getvalue(), str(e)

# Main Application
st.title("🏛️ NVIDIA NIM Professional Quant Station")

st.markdown("### 📥 Select Dataset")
colA, colB, colC = st.columns([1.5, 2, 1])
with colA:
    symbol_choice = st.selectbox(
        "Select Symbol", 
        ["XAUUSD", "EURUSD", "DXY"],
        label_visibility="collapsed"
    )
with colB:
    if st.button(f"🚀 Load {symbol_choice} Data", width="stretch"):
        st.session_state.current_symbol_view = symbol_choice
        source = st.session_state.get("data_source_priority", "☁️ Cloud Hub")
        
        load_success = False

        # --- BRANCH A: CLOUD HUB FIRST ---
        if "Cloud Hub" in source:
            if hf_repo and hf_token:
                from data_sync import load_from_hf
                with st.spinner(f"📥 Pulling {symbol_choice} from Cloud Hub..."):
                    try:
                        hub_df = load_from_hf(hf_repo, symbol_choice, hf_token)
                        st.session_state.df = hub_df
                        st.session_state[f"df_{symbol_choice}"] = hub_df
                        st.session_state.file_name = f"Cloud_{symbol_choice}_Data"
                        st.success(f"Latest {symbol_choice} data retrieved from Hub!")
                        load_success = True
                    except Exception as e:
                        st.warning(f"⚠️ Cloud sync failed: {e}")

        # --- BRANCH B: LOCAL DISK FIRST ---
        else:
            import os
            filename = f"{symbol_choice}_M1_Data.parquet"
            if os.path.exists(filename):
                with st.spinner(f"Loading {filename} from disk..."):
                    try:
                        st.session_state.df = pd.read_parquet(filename)
                        st.session_state.file_name = filename
                        load_success = True
                    except Exception as e:
                        st.warning(f"⚠️ Local file error: {e}")

        # --- FALLBACKS ---
        if not load_success:
            # If Cloud failed, try Local
            if "Cloud Hub" in source:
                import os
                filename = f"{symbol_choice}_M1_Data.parquet"
                if os.path.exists(filename):
                    st.session_state.df = pd.read_parquet(filename)
                    st.session_state.file_name = filename
                    load_success = True
            # If Local failed, try Cloud
            else:
                if hf_repo and hf_token:
                    from data_sync import load_from_hf
                    try:
                        hub_df = load_from_hf(hf_repo, symbol_choice, hf_token)
                        st.session_state.df = hub_df
                        st.session_state[f"df_{symbol_choice}"] = hub_df
                        st.session_state.file_name = f"Cloud_{symbol_choice}_Data"
                        load_success = True
                    except: pass

        if load_success:
            # --- 🚀 NEW: MASTER AUTO-BRIDGE ---
            current_df = st.session_state.df
            if current_df is not None and not current_df.empty:
                from data_sync import get_gap_info
                gap = get_gap_info(current_df)
                
                # If gap is larger than 1 hour, try to auto-bridge via Yahoo Finance
                if gap["gap_hours"] > 1.0 and symbol_choice in YAHOO_MAPPING:
                    target_yh = YAHOO_MAPPING[symbol_choice]
                    with st.status(f"🛰️ Auto-Bridging {symbol_choice}... (Gap: {gap['label']})"):
                        try:
                            from data_sync import sync_yahoo_symbol
                            # Use '1m' to match the M1 master history
                            # period '7d' is the max for 1m data on Yahoo Finance
                            sync_df, sync_stats = sync_yahoo_symbol(hf_repo, target_yh, hf_token, period="7d", interval="1m")
                            if sync_stats["status"] == "synced":
                                st.session_state.df = sync_df
                                st.session_state[f"df_{symbol_choice}"] = sync_df
                                st.toast(f"✅ Master Hub Healed! (+{sync_stats['new_rows']} minutes)")
                        except Exception as yh_err:
                            st.toast(f"ℹ️ Auto-Bridge skipped: {yh_err}")

            st.session_state.messages = []
            st.rerun()
        else:
            st.error(f"No data found for {symbol_choice} in Cloud OR Local Disk!")

with colC:
    if st.button("🧹 Clear Dataset", width="stretch"):
        if 'df' in st.session_state:
            del st.session_state['df']
        st.session_state.messages = []
        st.rerun()

# File uploader (Fallback)
uploaded_file = st.file_uploader("Or manually upload a CSV or Parquet dataset file", type=['csv', 'parquet'])

if uploaded_file is not None:
    if 'df' not in st.session_state or st.session_state.file_name != uploaded_file.name:
        with st.spinner("Optimizing data..."):
            st.session_state.df = process_data(uploaded_file)
            st.session_state.file_name = uploaded_file.name
            st.session_state.messages = []
            st.success("Professional environment initialized!")

if 'df' in st.session_state and st.session_state.df is not None:
    df_raw = st.session_state.df.copy()
    
    # --- DATE RANGE FILTERING ---
    time_cols = [c for c in df_raw.columns if 'time' in c.lower() or 'date' in c.lower()]
    if time_cols:
        main_time_col = time_cols[0]
        # Force to datetime for filtering
        df_raw[main_time_col] = pd.to_datetime(df_raw[main_time_col], errors='coerce')
        valid_dates = df_raw[main_time_col].dropna()
        if not valid_dates.empty:
            st.markdown("### 📅 Filter Data by Date")
            min_date = valid_dates.min().date()
            max_date = valid_dates.max().date()
            
            selected_dates = st.date_input(
                f"Select Range ({main_time_col})", 
                value=[min_date, max_date], 
                min_value=min_date, 
                max_value=max_date
            )
            
            if len(selected_dates) == 2:
                start_date, end_date = selected_dates
                mask = (df_raw[main_time_col].dt.date >= start_date) & (df_raw[main_time_col].dt.date <= end_date)
                df = df_raw.loc[mask].copy()
                st.info(f"✅ Filtered to {len(df):,} rows (from {start_date} to {end_date}).")
            else:
                df = df_raw.copy()
        else:
            df = df_raw.copy()
    else:
        df = df_raw.copy()
        
    # Convert time columns back to strings so Arrow Serialization never fails when rendering Tabs
    for c in df.columns:
        if str(df[c].dtype).startswith("datetime"):
            df[c] = df[c].astype(str)


    # TABBED NAVIGATION
    tab1, tab2, tab3 = st.tabs(["💬 AI Analyst", "🔍 Visual Explorer", "📋 Interactive Data Grid"])

    with tab1:
        st.subheader("🤖 AI Data Analyst")
        with st.expander("📊 Data Preview & Schema", expanded=True):
            st.dataframe(make_arrow_safe(df.head()), width="stretch")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Column Types:")
                # Avoid Arrow error: df.dtypes has dtype objects PyArrow can't serialize
                dtypes_df = pd.DataFrame({"Column": df.columns, "Type": [str(t) for t in df.dtypes]})
                st.dataframe(make_arrow_safe(dtypes_df), width="stretch", hide_index=True)
            with col2:
                st.write("Basic Stats:")
                st.dataframe(make_arrow_safe(df.describe()), width="stretch")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "code" in message:
                    with st.expander("View Code"): st.code(message["code"])
                    # Re-execute code on each render so charts/tables appear (Streamlit loses them on rerun)
                    stdout, error = execute_generated_code(message["code"], df)
                    if error:
                        st.error(f"⚠️ Error: {error}")
                    if stdout and stdout.strip():
                        st.markdown("**Output:**")
                        st.text(stdout)
                elif "exec_result" in message:
                    st.markdown(message["exec_result"])

        # --- QUANT PROMPT LIBRARY ---
        st.markdown("---")
        with st.expander("💡 **Quant Prompt Library** (One-Click Analysis)", expanded=False):
            st.info("Select a professional analysis template below to instantly generate a quantitative report.")
            
            st.markdown("##### 📌 Standard Analysis")
            prompt_cols = st.columns(2)
            with prompt_cols[0]:
                if st.button("📈 Target Escalation Matrix"):
                    st.session_state.pending_prompt = "Analyze moves that reach $10 (100 points). Calculate the probability of them extending to $15 and $20 using $2.5 bins before a 30% reversal occur. Show results in a clean table and a probability bar chart."
                if st.button("🌍 Session Volatility Profile"):
                    st.session_state.pending_prompt = "Break down the average 'DifferencePoints' and 'DifferencePercent' by trading session (CrossoverStartSession). Identify which session has the highest volatility and show a distribution plot formatted for a quant presentation."
            with prompt_cols[1]:
                if st.button("🔄 Mean Reversion Diagnostic"):
                    st.session_state.pending_prompt = "For all moves exceeding $12, what is the 'YES' vs 'NO' rate for 'ReversalTriggered'? Show the average time taken for a reversal (ReversalTime - AbsolutePeakTime) per session in a summary table."
                if st.button("⏱️ Wave Life-Cycle Analysis"):
                    st.session_state.pending_prompt = "Calculate the average time in minutes from CrossoverStartTime to AbsolutePeakTime for each session. Show a comparative horizontal bar chart and highlight the session where price peaks the fastest."

            st.markdown("---")
            st.markdown("##### 🔬 Deep Retracement Analysis (New)")
            deep_cols = st.columns(2)
            with deep_cols[0]:
                if st.button("🔻 Crossover Funnel Report"):
                    st.session_state.pending_prompt = "Build a conversion funnel analysis: First count total crossovers. Then count how many had DifferencePoints >= 10 (crossed the $10 threshold). Then count how many of those had ReversalTriggered = YES. Show as a funnel table AND a bar chart with counts and conversion % at each step. Break it down by CrossoverStartSession."
                if st.button("📊 True Depth Distribution"):
                    st.session_state.pending_prompt = "For all waves where ReversalTriggered is YES, calculate the TRUE retracement percentage using the formula: true_retrace_pct = (ImpulsePeakPrice - DeepestRetracePrice) / (ImpulsePeakPrice - CrossoverStartPrice) * 100. Bin these results into: 30-40%, 40-60%, 60-80%, 80-100%, and 100%+. Show as a stacked bar chart broken down by CrossoverStartSession. This tells us how deep the market ACTUALLY went beyond the 30% trigger."
            with deep_cols[1]:
                if st.button("🛡️ Shallow vs Deep Reversal Outcome"):
                    st.session_state.pending_prompt = "For all waves where ReversalTriggered is YES, calculate true_retrace_pct = (ImpulsePeakPrice - DeepestRetracePrice) / (ImpulsePeakPrice - CrossoverStartPrice) * 100. Split into Shallow (30-50%) and Deep (50%+). For each group, show: 1) How many had AbsolutePeakPrice > ImpulsePeakPrice (trend survived and made new high), 2) Average DifferencePoints, 3) Distribution by session. Show as a side-by-side comparison table and chart."
                
                if st.button("🎯 Session Target Range"):
                    st.session_state.pending_prompt = "Filter rows where ReversalTriggered is YES. For BULL calculate Move=ImpulsePeakPrice-CrossoverStartPrice, for BEAR calculate Move=CrossoverStartPrice-ImpulsePeakPrice. Show average, median, min, max Move grouped by CrossoverStartSession. This tells us typical extension after $10 crossover but BEFORE the 30% reversal trigger. Show as a distribution box plot by session."

        # Handle button clicks from library
        prompt = st.chat_input("Ask for analysis, code, or charts...")
        if "pending_prompt" in st.session_state:
            prompt = st.session_state.pending_prompt
            del st.session_state.pending_prompt

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)

            if not api_key_to_use: st.warning("Missing API Key. Please enter your key in the sidebar.")
            else:
                with st.chat_message("assistant"):
                    df_info = io.StringIO()
                    df.info(buf=df_info)
                    metadata = df_info.getvalue()
                    
                    system_prompt = f"""
                    You are a Lead Quant Analyst in 2026. Use the provided DataFrame 'df' to answer the user's question.
                    
                    TOOLS AVAILABLE:
                    - pandas_ta (as ta): Technical indicators.
                    - quantstats (as qs): Financial performance reports.
                    - statsmodels (as sm): Econometric tests.
                    - pygwalker (as pyg): For visual explorers (use pyg.walk(df)).
                    - math, scipy, sklearn, numpy, polars (as pl), duckdb (as duck), xgboost (as xgb).
                    
                    DATAFRAME SCHEMA:
                    {metadata}
                    
                    DATA SAMPLES:
                    {df.head(3).to_string()}
                    
                    RULES:
                    1. Provide ONLY executable Python code within ```python blocks for analysis.
                    2. Use 'st.write()', 'st.pyplot()', or 'st.plotly_chart()' for visual outputs.
                    3. Assume 'df' is already defined.
                    4. IMPORTANT: Always explain your reasoning BEFORE the code block.
                    """
                    
                    if show_debug:
                        with st.expander("🔍 RAW AI PROMPT", expanded=False):
                            st.json({"role": "system", "content": system_prompt})
                            st.json(st.session_state.messages)
                    
                    def get_ai_response(msgs):
                        full_txt = ""
                        ph = st.empty()
                        try:
                            # Robust key handling
                            final_key = api_key_to_use
                            if provider_choice == "NVIDIA" and not final_key.startswith("nvapi-"):
                                final_key = f"nvapi-{final_key}"
                            
                            client = OpenAI(base_url=base_url, api_key=final_key)
                            print(f"DEBUG: Starting stream with model {model_choice}...")
                            
                            extra_kwargs = {}
                            if provider_choice == "NVIDIA" and enable_thinking:
                                extra_kwargs = {"chat_template_kwargs": {"enable_thinking": True}}

                            stream = client.chat.completions.create(
                                model=model_choice,
                                messages=msgs,
                                temperature=0.2,
                                top_p=0.7,
                                max_tokens=16384,
                                stream=True,
                                extra_body=extra_kwargs if extra_kwargs else None
                            )
                            
                            for chunk in stream:
                                if chunk.choices[0].delta.content is not None:
                                    delta = chunk.choices[0].delta.content
                                    full_txt += delta
                                    ph.markdown(full_txt + "▌")
                            
                            ph.markdown(full_txt)
                            print(f"DEBUG: Stream finished. Length: {len(full_txt)}")
                            return full_txt
                        except Exception as e:
                            print(f"DEBUG: OpenAI Client Error: {e}")
                            st.error(f"API Error: {e}")
                            return f"ERROR: {e}"

                    # Initial Chat
                    current_msgs = [{"role": "system", "content": system_prompt}] + st.session_state.messages
                    
                    with st.spinner("AI Analyst is thinking..."):
                        ai_text = get_ai_response(current_msgs)
                    
                    if show_debug:
                        with st.expander("🔍 RAW AI RESPONSE", expanded=False):
                            st.text(ai_text)
                    
                    # Self-Healing Loop (Max 2 attempts)
                    final_code = None
                    final_stdout = ""
                    
                    for attempt in range(2):
                        # Extracting ALL code blocks
                        code_pattern = r"```python(.*?)```"
                        blocks = re.findall(code_pattern, ai_text, re.S | re.I)
                        
                        if blocks:
                            full_code_to_run = "\n\n".join([b.strip() for b in blocks])
                            print(f"DEBUG: Code extracted. Attempt {attempt+1}")
                            stdout, error = execute_generated_code(full_code_to_run, df)
                            
                            if error:
                                print(f"DEBUG: Execution Error: {error}")
                                st.warning(f"⚠️ Attempt {attempt+1} failed with error: {error}. Auto-healing...")
                                heal_prompt = f"The previous code failed with this error: {error}. Please fix the code and try again. Provide ONLY the corrected code."
                                current_msgs.append({"role": "assistant", "content": ai_text})
                                current_msgs.append({"role": "user", "content": heal_prompt})
                                with st.spinner(f"AI is repairing code (Attempt {attempt+2})..."):
                                    ai_text = get_ai_response(current_msgs)
                            else:
                                print(f"DEBUG: Execution Successful!")
                                final_code = full_code_to_run
                                final_stdout = stdout
                                break
                        else:
                            print("DEBUG: No code blocks found in AI response.")
                            break
                    
                    # Store final result
                    msg_to_store = {"role": "assistant", "content": ai_text}
                    if final_code:
                        msg_to_store["code"] = final_code
                    if final_stdout:
                        msg_to_store["exec_result"] = final_stdout
                    
                    st.session_state.messages.append(msg_to_store)
                    st.rerun()

    with tab2:
        st.subheader("📊 Drag-and-Drop Visual Explorer")
        if PYG_AVAILABLE:
            # Initialize PyGWalker (use Arrow-safe copy)
            pyg_html = pyg.to_html(make_arrow_safe(df))
            st.components.v1.html(pyg_html, height=1000, scrolling=True)
        else:
            st.error("PyGWalker is unavailable due to memory constraints on this system.")
            st.info("You can still use the AI Analyst tab to generate charts via Plotly/Matplotlib.")

    with tab3:
        st.subheader("🗃️ Interactive Data Grid")
        if AGGRID_AVAILABLE:
            AgGrid(make_arrow_safe(df), height=600, theme='alpine', enable_enterprise_modules=False)
        else:
            st.error("AgGrid is unavailable due to memory constraints.")
            st.dataframe(make_arrow_safe(df.head(100)))

else:
    st.write("👋 Welcome! Please upload your dataset (CSV or Parquet) to start the analysis.")
    st.info("💡 **Any time-series, financial OHLC data, or custom data is completely supported.** The AI knows how to adapt to your specific dataset.")
    st.image("https://developer.nvidia.com/sites/default/files/akamai/NVIDIA_NIM_Icon.png", width=100)
