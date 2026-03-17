
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

# Page configuration
st.set_page_config(page_title="NVIDIA Quant Insight Bot", layout="wide")

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
    
    if st.session_state.authenticated:
        return True
    
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
        
        if st.button("📥 Download EA Package (ZIP)", use_container_width=True):
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
        if st.button("🚪 Logout", use_container_width=True):
            logout()
    
    return True

# Check authentication before showing app
if not check_password():
    st.stop()

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    # Leave value empty to prevent dots being shown in the DOM (security measure)
    input_api_key = st.text_input("NVIDIA API Key", value="", type="password", help="Enter your NVIDIA NIM API Key. If left blank, the system will attempt to use the pre-configured station key.")
    
    # Resolve the key: User input takes priority, secret is used as a silent fallback
    api_key_to_use = input_api_key if input_api_key else st.secrets.get("NVIDIA_API_KEY", "")
    
    model_choice = st.selectbox("Select Model", [
        "qwen/qwen3.5-122b-a10b",
        "qwen/qwen2.5-coder-32b-instruct",
        "deepseek-ai/deepseek-v3.1",
        "deepseek-ai/deepseek-r1-distill-qwen-32b",
        "nvidia/llama-3.1-405b-instruct"
    ])
    enable_thinking = st.checkbox("Enable Thinking Mode", value=True)
    show_debug = st.checkbox("Show AI Trace (Debug)", value=False)
    
    st.markdown("---")
    st.subheader("🕵️ AI Connection Diagnostic")
    if st.button("Run Connection Test"):
        if not api_key_to_use:
            st.error("No API Key found! Please enter one.")
        else:
            try:
                # Test with standard prefix logic
                test_key = api_key_to_use if api_key_to_use.startswith("nvapi-") else f"nvapi-{api_key_to_use}"
                test_client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=test_key)
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
    
    st.info("Upload your CSV file to begin analysis.")

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
        # Load EVERYTHING as string first to prevent 'object' type confusion
        df = pd.read_csv(uploaded_file, dtype=str, low_memory=False)
        
        # Manually convert known numeric columns
        numeric_cols = [
            'cross_price', 'segment_index', 'segment_price', 
            'segment_size_price', 'segment_move_points', 
            'segment_move_percent', 'sequence_extreme_price', 'is_final'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ensure 'symbol' and 'session' are clean strings
        for col in ['symbol', 'session', 'cross_type', 'segment_direction']:
            if col in df.columns:
                df[col] = df[col].fillna("UNKNOWN").astype(str)
        
        # Convert timestamps (naive datetime64[ns] for Arrow compatibility)
        time_cols = ['cross_time', 'cross_end_time', 'segment_time', 'sequence_extreme_time']
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
        'alt': alt
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

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

if uploaded_file is not None:
    if 'df' not in st.session_state or st.session_state.file_name != uploaded_file.name:
        with st.spinner("Optimizing data with Parquet..."):
            st.session_state.df = process_data(uploaded_file)
            st.session_state.file_name = uploaded_file.name
            st.session_state.messages = []
            st.success("Professional environment initialized!")

    df = st.session_state.df

    # TABBED NAVIGATION
    tab1, tab2, tab3 = st.tabs(["💬 AI Analyst", "🔍 Visual Explorer", "📋 Interactive Data Grid"])

    with tab1:
        st.subheader("🤖 AI Data Analyst")
        with st.expander("📊 Data Preview & Schema", expanded=True):
            st.dataframe(make_arrow_safe(df.head()), use_container_width=True)
            col1, col2 = st.columns(2)
            with col1:
                st.write("Column Types:")
                # Avoid Arrow error: df.dtypes has dtype objects PyArrow can't serialize
                dtypes_df = pd.DataFrame({"Column": df.columns, "Type": [str(t) for t in df.dtypes]})
                st.dataframe(make_arrow_safe(dtypes_df), use_container_width=True, hide_index=True)
            with col2:
                st.write("Basic Stats:")
                st.dataframe(make_arrow_safe(df.describe()), use_container_width=True)

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

        if prompt := st.chat_input("Ask for analysis, code, or charts..."):
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
                    - math, scipy, sklearn, numpy, polars (as pl), duckdb (as duck).
                    
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
                            # Robust key handling: NVIDIA keys often require the nvapi- prefix
                            final_key = api_key_to_use if api_key_to_use.startswith("nvapi-") else f"nvapi-{api_key_to_use}"
                            
                            # Using integrate.api.nvidia.com as the base
                            client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=final_key)
                            print(f"DEBUG: Starting stream with model {model_choice}...")
                            
                            stream = client.chat.completions.create(
                                model=model_choice,
                                messages=msgs,
                                temperature=0.2,
                                top_p=0.7,
                                max_tokens=16384,
                                stream=True,
                                extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}}
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
                            st.error(f"NVIDIA API Error: {e}")
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
    st.write("👋 Welcome! Please upload a crossover impulse dataset to start the analysis.")
    st.image("https://developer.nvidia.com/sites/default/files/akamai/NVIDIA_NIM_Icon.png", width=100)
