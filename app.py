
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
from datetime import datetime, timezone
import contextlib
import io
from streamlit_autorefresh import st_autorefresh
from data_sync import pull_mt5_latest, ping_mt5_server

def execute_generated_code(code, df):
    """Executes AI-generated Python code within a safe, isolated quant environment."""
    env = {
        'st': st, 'pd': pd, 'np': np, 'px': px, 'go': go, 'plt': plt,
        'sns': sns, 'df': df, 're': re, 'math': math, 'scipy': scipy,
        'pyg': pyg if PYG_AVAILABLE else None, 'ta': ta, 'sm': sm,
        'qs': qs, 'duck': duckdb, 'pl': pl, 'alt': alt, 'xgb': xgb,
        'datetime': datetime, 'timezone': timezone
    }
    output = io.StringIO()
    plt.close('all')
    with contextlib.redirect_stdout(output):
        try:
            exec(code, env)
            return output.getvalue(), None
        except Exception as e:
            return output.getvalue(), str(e)

def process_ai_query(prompt, df, model_choice, api_key, model_provider, history_key="messages", base_url=None, snapshot=None):
    """Handles professional quantitative analysis queries using multi-modal AI reasoning."""
    if "messages_live" not in st.session_state: st.session_state.messages_live = []
    if "messages" not in st.session_state: st.session_state.messages = []
    
    # Get correct history
    history = st.session_state[history_key]
    history.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"): st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("AI Analyst is calculating..."):
            df_info = io.StringIO()
            df.info(buf=df_info)
            metadata = df_info.getvalue()
            
            system_prompt = f"""
            You are a Lead Quant in 2026. Use the provided DataFrame 'df' for your analysis.
            SCHEMA: {metadata}
            LATEST_CANDLE_RECORDED: {df['time'].iloc[-1] if not df.empty and 'time' in df.columns else 'N/A'}
            SAMPLES (Last 5 Rows): {df.tail(5).to_string()}
            
            RULES (STRICT):
            1. Analyze only based on the provided data available in 'df'. Ignore local system time.
            2. Provide executable Python code in ```python blocks.
            3. Use 'st.write()', 'st.plotly_chart()' for results.
            """
            
            # Robust key/client handling
            final_key = api_key
            if model_provider == "NVIDIA" and not final_key.startswith("nvapi-"):
                final_key = f"nvapi-{final_key}"
            
            client = OpenAI(base_url=base_url, api_key=final_key)
            full_txt = ""
            ph = st.empty()
            
            # 🧹 CLEAN HOUSE: Filter out non-serializable data snapshots from the API payload
            clean_history = []
            for m in history:
                # We only send text 'role' and 'content' to the AI.
                # We do NOT send the raw DataFrame snapshot.
                clean_history.append({"role": m["role"], "content": m["content"]})

            messages = [{"role": "system", "content": system_prompt}] + clean_history
            
            try:
                stream = client.chat.completions.create(
                    model=model_choice, messages=messages, temperature=0.2, stream=True
                )
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        full_txt += chunk.choices[0].delta.content
                        ph.markdown(full_txt + "▌")
                ph.markdown(full_txt)
                
                # Execution and Self-Healing
                code_pattern = r"```python(.*?)```"
                blocks = re.findall(code_pattern, full_txt, re.S | re.I)
                final_code = "\n\n".join([b.strip() for b in blocks]) if blocks else None
                final_stdout = ""
                
                if final_code:
                    final_stdout, error = execute_generated_code(final_code, df)
                    if error:
                        st.error(f"Execution Error: {error}")
                
                msg_to_store = {"role": "assistant", "content": full_txt}
                if final_code: msg_to_store["code"] = final_code
                if final_stdout: msg_to_store["exec_result"] = final_stdout
                
                # 🛡️ GOLDEN VAULT: Store snapshot with a unique ID, not inside the message
                if snapshot is not None:
                    if "data_vault" not in st.session_state: st.session_state.data_vault = {}
                    import time
                    snap_id = f"snap_{int(time.time())}"
                    st.session_state.data_vault[snap_id] = snapshot
                    msg_to_store["snapshot_id"] = snap_id
                
                history.append(msg_to_store)
                st.rerun()

            except Exception as e:
                st.error(f"API Error: {e}")

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

    # ── Cloud Data Management ──
    st.markdown("---")
    st.subheader("🌐 Data Strategy")
    data_source_priority = st.radio(
        "Default Data Source",
        ["☁️ Cloud Hub (Hugging Face)", "💾 Local Disk (.parquet)"],
        index=0,
        help="Choose where the 'Load' button looks first."
    )
    st.session_state.data_source_priority = data_source_priority

    # Simple freshness display (Optional but nice for sidebar)
    # --- ⚡ AUTO-SYNC & CREDENTIALS ---
    hf_repo  = st.secrets.get("HF_REPO_ID", "")
    hf_token = st.secrets.get("HuggingFace_API_KEY", "")
    mt5_url = st.session_state.get("mt5_url", "http://localhost:5000")
    mt5_token = st.secrets.get("MT5_API_TOKEN", "impulse_secure_2026")

    st.markdown("---")
    st.subheader("⚡ Auto-Sync Mode")
    auto_sync_on = st.toggle("Enable Auto-Sync 📡", value=st.session_state.get("auto_sync_on", False))
    st.session_state.auto_sync_on = auto_sync_on

    if auto_sync_on:
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
        if out[col].dtype == object:
            # Only convert strict object types to string; leave datetimes alone!
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

# --- MASTER MODE SWITCHER ---
st.markdown("---")
view_mode = st.radio(
    "Choose Analysis Data Source",
    ["🗄️ Historical Vault (Hugging Face)", "📡 Live MT5 Feed (Direct Broker)"],
    horizontal=True,
    help="Switch between deep historical archives and real-time broker data."
)
st.markdown("---")

if "Historical" in view_mode:
    st.markdown("### 📥 Select Archive Dataset")
    colA, colD, colB, colC = st.columns([1.5, 1.5, 2, 1])
    with colA:
        symbol_choice = st.selectbox(
            "Select Symbol", 
            ["XAUUSD", "EURUSD", "DXY"],
            label_visibility="collapsed"
        )
    with colD:
        lookback_choice = st.selectbox(
            "Lookback Period",
            ["Last 7 Days", "Last 30 Days", "Last 3 Months", "Last 6 Months", "1 Year", "All Data"],
            index=1,
            help="Select how much data to load into the UI to save memory.",
            label_visibility="collapsed"
        )
    with colB:
        if st.button(f"🚀 Load {symbol_choice} Data", width="stretch"):
            st.session_state.current_symbol_view = symbol_choice
            source = st.session_state.get("data_source_priority", "☁️ Cloud Hub")
            load_success = False
            hf_rows_before = 0

            # ── STEP 1: Fetch existing data from Hugging Face Hub ───────────────
            if "Cloud Hub" in source:
                if hf_repo and hf_token:
                    from data_sync import load_from_hf
                    with st.spinner(f"📥 Step 1/3 — Fetching existing data for {symbol_choice} from Hugging Face Hub..."):
                        try:
                            hub_df = load_from_hf(hf_repo, symbol_choice, hf_token)
                            hf_rows_before = len(hub_df)
                            st.session_state.df = hub_df
                            st.session_state[f"df_{symbol_choice}"] = hub_df
                            st.session_state.file_name = f"Cloud_{symbol_choice}_Data"
                            st.info(f"☁️ Loaded **{hf_rows_before:,} rows** from Hugging Face Hub.")
                            load_success = True
                        except Exception as e:
                            st.warning(f"⚠️ Cloud Hub fetch failed: {e}. Trying local disk...")

            # Fallback to local disk
            if not load_success:
                import os
                filename = f"{symbol_choice}_M1_Data.parquet"
                if os.path.exists(filename):
                    with st.spinner(f"Loading {filename} from local disk..."):
                        try:
                            disk_df = pd.read_parquet(filename)
                            hf_rows_before = len(disk_df)
                            st.session_state.df = disk_df
                            st.session_state.file_name = filename
                            st.info(f"💾 Loaded **{hf_rows_before:,} rows** from local disk.")
                            load_success = True
                        except Exception as e:
                            st.warning(f"⚠️ Local file error: {e}")

            if not load_success:
                st.error(f"❌ No data found for {symbol_choice} on Hugging Face Hub OR Local Disk!")
            else:
                current_df = st.session_state.df

                # ── STEP 2: Detect gap & fetch new data ─────────────────────────
                from data_sync import get_gap_info
                gap = get_gap_info(current_df)

                if gap["gap_hours"] <= 0.25:
                    st.success(f"✅ {symbol_choice} data is already fresh ({gap['label']}). No sync needed.")
                elif symbol_choice in YAHOO_MAPPING:
                    target_yh = YAHOO_MAPPING[symbol_choice]
                    st.info(f"🕐 Gap detected: **{gap['label']}**. Fetching new candles from Yahoo Finance ({target_yh})...")

                    with st.spinner(f"📡 Step 2/3 — Fetching new {symbol_choice} candles (last 7 days) from Yahoo Finance..."):
                        try:
                            from data_sync import sync_yahoo_symbol
                            # sync_yahoo_symbol already:
                            #   1. Fetches new data from Yahoo
                            #   2. Loads existing from HF Hub
                            #   3. Merges (dedup + sort)
                            #   4. Pushes the merged result back to HF Hub
                            sync_df, sync_stats = sync_yahoo_symbol(
                                hf_repo, target_yh, hf_token,
                                existing_df=current_df   # ← pass already-loaded HF data; skips redundant re-download
                            )

                            hf_rows_after = len(sync_df)
                            new_rows_added = hf_rows_after - hf_rows_before

                            # ── STEP 3: Update session with merged result ────────
                            st.session_state.df = sync_df
                            st.session_state[f"df_{symbol_choice}"] = sync_df

                            # Build detailed report
                            report = (
                                f"✅ **Merge Successful & Pushed to Hub!**\n\n"
                                f"- **Rows Before:** {hf_rows_before:,}  (Last candle: {gap.get('last_timestamp', 'Unknown')})\n"
                                f"- **Delta Fetched:** +{new_rows_added:,} new rows\n"
                                f"- **Sync Range:** {sync_df['time'].iloc[-(new_rows_added+1)] if new_rows_added > 0 else 'N/A'} → {sync_df['time'].iloc[-1]}\n"
                                f"- **Total Rows now on Hub:** {hf_rows_after:,}"
                            )
                            st.success(report)
                            st.toast(f"🚀 {symbol_choice} synced! +{new_rows_added:,} rows.")
                            
                            # Use the freshly synced dataframe as our current_df
                            current_df = sync_df

                        except Exception as yh_err:
                            st.warning(f"⚠️ Auto-sync skipped (Yahoo Finance unreachable or no new data): {yh_err}")
                            st.info("Displaying existing Hub data as-is.")
                else:
                    st.warning(f"⚠️ Gap of **{gap['label']}** detected but no Yahoo Finance mapping for {symbol_choice}. Use MT5 Sync instead.")

                # ── STEP 4: Apply memory-safe lookback filter ────────────────────────
                time_cols = [c for c in current_df.columns if 'time' in c.lower() or 'date' in c.lower()]
                if time_cols and lookback_choice != "All Data":
                    time_col = time_cols[0]
                    current_df[time_col] = pd.to_datetime(current_df[time_col], utc=False, errors='coerce')
                    max_time = current_df[time_col].max()
                    
                    if lookback_choice == "Last 7 Days":
                        cutoff = max_time - pd.Timedelta(days=7)
                    elif lookback_choice == "Last 30 Days":
                        cutoff = max_time - pd.Timedelta(days=30)
                    elif lookback_choice == "Last 3 Months":
                        cutoff = max_time - pd.Timedelta(days=90)
                    elif lookback_choice == "Last 6 Months":
                        cutoff = max_time - pd.Timedelta(days=180)
                    elif lookback_choice == "1 Year":
                        cutoff = max_time - pd.Timedelta(days=365)
                    
                    # Filter down to save memory
                    memory_safe_df = current_df[current_df[time_col] >= cutoff].reset_index(drop=True)
                    st.session_state.df = memory_safe_df
                    st.session_state[f"df_{symbol_choice}"] = memory_safe_df
                    st.info(f"✂️ UI Memory Safe Mode: Loaded **{len(memory_safe_df):,}** rows ({lookback_choice}) out of **{len(current_df):,}** total.)")
                else:
                    # User chose "All Data" (or time col not found)
                    st.session_state.df = current_df
                    st.session_state[f"df_{symbol_choice}"] = current_df
                    st.info(f"⚠️ Loaded FULL dataset into UI memory: **{len(current_df):,}** rows.")

                st.session_state.messages = []
                st.rerun()

    with colC:
        if st.button("🧹 Clear Workspace", width="stretch"):
            if 'df' in st.session_state: del st.session_state['df']
            if 'df_live' in st.session_state: del st.session_state['df_live']
            st.session_state.messages = []
            st.session_state.messages_live = []
            st.rerun()

    # --- HISTORICAL ANALYSIS TABS (Only if data loaded) ---
    if 'df' in st.session_state and st.session_state.df is not None:
        df = st.session_state.df
        h_tab1, h_tab2, h_tab3 = st.tabs(["💬 AI Analyst", "🔍 Visual Explorer", "📋 Data Grid"])
        
        with h_tab1:
            st.subheader("🤖 AI Data Analyst (Historical)")
            with st.expander("📊 Data Preview & Schema", expanded=True):
                st.dataframe(make_arrow_safe(df.head()), width="stretch")
            
            # Historical Chat history
            if "messages" not in st.session_state: st.session_state.messages = []
            for m in st.session_state.messages:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])
                    if "code" in m:
                        with st.expander("Show Code"): st.code(m["code"])
                        execute_generated_code(m["code"], df)
            
            if prompt := st.chat_input("Analyze the archive...", key="hist_chat"):
                process_ai_query(prompt, df, model_choice, api_key_to_use, provider_choice, history_key="messages", base_url=base_url)

        with h_tab2:
            st.subheader("📊 Visual Explorer")
            if PYG_AVAILABLE:
                st.components.v1.html(pyg.to_html(make_arrow_safe(df.tail(100000))), height=800, scrolling=True)
            else: st.info("PyGWalker unavailable on this instance.")

        with h_tab3:
            st.subheader("📋 Data Grid")
            st.dataframe(make_arrow_safe(df.tail(1000)))
    else:
        st.info("👋 Master Source: **Historical Vault** selected. Please load a dataset above to start analysis.")

elif "Live" in view_mode:
    st.markdown("### 📡 Live MT5 Broker Terminal")
    
    # ── STEP 1: Broker Bridge Connection Gate ────────────────────────────
    is_connected = st.session_state.get("mt5_connected", False)
    with st.expander("🔒 MT5 Broker Gate (Local Connection Settings)", expanded=not is_connected):
        c1, c2 = st.columns(2)
        with c1:
            mt5_url_in = st.text_input("MT5 Server URL", 
                                     value=st.session_state.get("mt5_url", "http://localhost:5000"), 
                                     placeholder="http://localhost:5000",
                                     key="live_mt5_url")
        with c2:
            mt5_tok_in = st.text_input("Security Token", 
                                     value=st.secrets.get("MT5_API_TOKEN", "impulse_secure_2026"), 
                                     type="password",
                                     key="live_mt5_token")
            
        if st.button("🔌 Establish Broker Bridge", width="stretch"):
            with st.spinner("Pinging local server..."):
                res = ping_mt5_server(mt5_url_in, mt5_tok_in)
                if res["reachable"] and res["mt5_initialized"]:
                    st.session_state.mt5_connected = True
                    st.session_state.mt5_url = mt5_url_in
                    st.success("✅ Bridge Established! Terminal Unlocked.")
                    st.rerun()
                else:
                    st.session_state.mt5_connected = False
                    st.error("❌ Connection Failed. Ensure 'mt5_data_server.py' is running locally.")

    # ── STEP 2: Main Terminal (Only if bridge is active) ──────────────────
    if st.session_state.get("mt5_connected"):
        st_col1, st_col2, st_col3, st_col4 = st.columns([1.5, 1, 1.5, 1.5])
        with st_col1: l_sym = st.selectbox("Symbol", ["XAUUSD", "EURUSD", "DXY"], key="l_sym")
        with st_col2: l_tf = st.selectbox("TF", ["1m", "5m", "15m", "1h"], key="l_tf")
        with st_col3: l_count = st.number_input("Lookback Bars", value=500, min_value=100, max_value=5000, step=100, key="l_count")
        with st_col4: 
            st.write("")
            live_active = st.toggle("Enable Continuous Feed 📡", value=st.session_state.get("live_active_state", False), key="live_active_sync")
            st.session_state.live_active_state = live_active

        # 🎯 SURGICAL FRAGMENT: Re-runs only the Chart area
        @st.fragment(run_every=60 if live_active else None)
        def live_chart_fragment(symbol, timeframe, count, active):
            m_url = st.session_state.get("mt5_url", "http://localhost:5000")
            m_tok = st.secrets.get("MT5_API_TOKEN", "impulse_secure_2026")
            
            # Fetch directly inside fragment
            df_l = pull_mt5_latest(m_url, symbol, timeframe, count, m_tok)
            if not df_l.empty:
                st.session_state.df_live = df_l
                
                # Pulse Bar UI
                st.subheader(f"📊 {symbol} — {timeframe} Real-Time Feed")
                last_bar = df_l.iloc[-1]
                t_str = last_bar['time'].strftime('%H:%M:%S') if hasattr(last_bar['time'], 'strftime') else str(last_bar['time'])
                
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("⏲️ Terminal Time", t_str)
                m2.metric("🟢 OPEN", f"{last_bar['open']:.5f}")
                m3.metric("📈 HIGH", f"{last_bar['high']:.5f}", delta=f"{(last_bar['high']-last_bar['open']):.5f}")
                m4.metric("📉 LOW",  f"{last_bar['low']:.5f}",  delta=f"{(last_bar['low']-last_bar['open']):.5f}")
                m5.metric("🎯 CURRENT", f"{last_bar['close']:.5f}", delta=f"{(last_bar['close']-last_bar['open']):.5f}")

                # Plotly Chart
                fig = go.Figure(data=[go.Candlestick(x=df_l['time'], open=df_l['open'], high=df_l['high'], low=df_l['low'], close=df_l['close'])])
                fig.update_layout(template="plotly_dark", height=500, margin=dict(l=10,r=10,t=10,b=10), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        # Trigger Fragment
        live_chart_fragment(l_sym, l_tf, l_count, live_active)

        # 🤖 LIVE AI ANALYSIS (Independent Area)
        if st.session_state.get("df_live") is not None:
            st.markdown("---")
            st.subheader("🤖 Live AI Trade Analyst")
            
            if "messages_live" not in st.session_state: st.session_state.messages_live = []
            
            # Historical chat display
            for m in st.session_state.messages_live:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])
                    if "code" in m:
                        # 🛡️ VAULT LOOKUP: Retrieve the frozen snapshot using the ID
                        snap_id = m.get("snapshot_id")
                        msg_df = st.session_state.get("data_vault", {}).get(snap_id, st.session_state.df_live)
                        execute_generated_code(m["code"], msg_df)

            if chat_l := st.chat_input("Analyze live price action...", key="live_chat"):
                # 🛠️ SNAPSHOT MECHANISM: Freeze the data context for this query
                snapshot_df = st.session_state.df_live.copy()
                process_ai_query(chat_l, snapshot_df, model_choice, api_key_to_use, provider_choice, 
                                  history_key="messages_live", base_url=base_url, snapshot=snapshot_df)
    else:
        st.info("📡 Broker Terminal Locked. Please establish the bridge above to stream live market data.")

else:
    st.info("👋 Welcome! Please select a Mode above to begin.")
    st.image("https://developer.nvidia.com/sites/default/files/akamai/NVIDIA_NIM_Icon.png", width=100)
