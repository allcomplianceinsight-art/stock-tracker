import streamlit as st
import yfinance as yf
import pandas as pd
import os
import time
import ta
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import requests
import plotly.graph_objects as go

st.set_page_config(page_title="Stock Tracker-A Product of Compliance In Sight", layout="wide")

# =========================
# THEME SWITCHER
# =========================

themes = {
    "Emerald Green": """
        <style>
        .stApp {
            background: linear-gradient(135deg, #064e3b 0%, #059669 50%, #34d399 100%);
            color: #f0fdf4 !important;
            font-family: 'Segoe UI', Roboto, sans-serif;        
        }
        </style>
    """,
    "Dark Neon": """
        <style>
        .stApp {
            background: linear-gradient(135deg, #000000 0%, #0f172a 40%, #1e3a8a 100%);
            color: #f8fafc !important;
            font-family: 'Orbitron', sans-serif;
        }
        </style>
    """,
    "Light Mode": """
        <style>
        .stApp {
            background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 50%, #e2e8f0 100%);
            color: #1e293b !important;
            font-family: 'Segoe UI', Roboto, sans-serif;
        }
        </style>
    """,
    "Sunset Gradient": """
        <style>
        .stApp {
            background: linear-gradient(135deg, #1a2a6c 0%, #b21f1f 50%, #fdbb2d 100%);
            color: #f8fafc !important;
            font-family: 'Segoe UI', Roboto, sans-serif;
        }
        </style>
    """,
    "Royal Purple": """
        <style>
        .stApp {
            background: linear-gradient(135deg, #3b0764 0%, #9333ea 50%, #c084fc 100%);
            color: #fdf4ff !important;
            font-family: 'Poppins', sans-serif;
        }
        </style>
    """,
    "Ocean Breeze": """
        <style>
        .stApp {
            background: linear-gradient(135deg, #0ea5e9 0%, #38bdf8 50%, #bae6fd 100%);
            color: #0c4a6e !important;
            font-family: 'Segoe UI', sans-serif;
        }
        </style>
    """,
    "Rose Gold": """
        <style>
        .stApp {
            background: linear-gradient(135deg, #fce7f3 0%, #f9a8d4 50%, #f472b6 100%);
            color: #831843 !important;
            font-family: 'Poppins', sans-serif;
        }
        </style>
    """,
    "Midnight Blue": """
        <style>
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
            color: #e2e8f0 !important;
            font-family: 'Roboto Mono', monospace;
        }
        </style>
    """,
    "Golden Hour": """
        <style>
        .stApp {
            background: linear-gradient(135deg, #facc15 0%, #f97316 50%, #b91c1c 100%);
            color: #1c1917 !important;
            font-family: 'Segoe UI', sans-serif;
        }
        </style>
    """,
    "Mint Fresh": """
        <style>
        .stApp {
            background: linear-gradient(135deg, #ecfdf5 0%, #6ee7b7 50%, #10b981 100%);
            color: #064e3b !important;
            font-family: 'Poppins', sans-serif;
        }
        </style>
    """,
    "Galaxy": """
        <style>
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #4c1d95 40%, #7e22ce 100%);
            color: #e9d5ff !important;
            font-family: 'Orbitron', sans-serif;
        }
        </style>
    """,
    "Candy Pop": """
        <style>
        .stApp {
            background: linear-gradient(135deg, #f87171 0%, #facc15 50%, #34d399 100%);
            color: #1f2937 !important;
            font-family: 'Comic Sans MS', cursive, sans-serif;
        }
        </style>
    """
}

# User chooses theme
selected_theme = st.sidebar.selectbox("🎨 Choose Theme", list(themes.keys()))

# Apply selected theme
st.markdown(themes[selected_theme], unsafe_allow_html=True)

# Your existing custom CSS (titles, buttons, expanders, etc.)

st.markdown(
    """
    <style>
    
    /* ===== Titles ===== */
    h1, h2, h3 {
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    h1 {
        color: #facc15;  /* gold accent */
        text-align: center;
        padding: 0.5rem;
    }
    h2 {
        color: #38bdf8; /* sky blue */
    }
    h3 {
        color: #c084fc; /* violet */
    }

    /* ===== DataFrame Styling ===== */
    .stDataFrame {
        border-radius: 14px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.6);
        background-color: #1e293b !important;
    }

    /* ===== Buttons ===== */
    button[kind="primary"] {
        background: linear-gradient(90deg, #2563eb, #1e40af);
        color: white !important;
        border-radius: 12px !important;
        font-weight: bold !important;
        padding: 8px 18px !important;
        transition: all 0.2s ease-in-out;
        border: none;
    }
    button[kind="primary"]:hover {
        background: linear-gradient(90deg, #1d4ed8, #1e3a8a);
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.6);
    }

    /* ===== Expanders ===== */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 16px;
        background: linear-gradient(90deg, #334155, #1e293b);
        color: #f1f5f9 !important;
        border-radius: 10px;
        padding: 10px;
    }
    .streamlit-expanderContent {
        background: #1e293b;
        padding: 12px;
        border-radius: 10px;
        box-shadow: inset 0 1px 4px rgba(0,0,0,0.4);
    }

    /* ===== Markdown Cards ===== */
    .stMarkdown {
        background: #1e293b;
        padding: 12px;
        border-radius: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.5);
        margin-bottom: 12px;
    }

    /* ===== Download Button ===== */
    .stDownloadButton>button {
        background: linear-gradient(90deg, #22c55e, #15803d);
        color: white !important;
        border-radius: 10px !important;
        font-weight: bold !important;
        padding: 8px 16px !important;
        transition: all 0.2s ease-in-out;
        border: none;
    }
    .stDownloadButton>button:hover {
        background: linear-gradient(90deg, #16a34a, #14532d);
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.5);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;700&display=swap');

    .custom-title {
        font-family: 'Orbitron', sans-serif !important;
        font-size: 1.2rem;
        font-weight: 600;
        background: linear-gradient(90deg, #facc15, #38bdf8, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        text-align: center;
    }

    .custom-box {
        text-align: center; 
        padding: 13px; 
        border-radius: 11px; 
        background: #1e293b; 
        box-shadow: 0 4px 12px rgba(0,0,0,0.6); 
        margin-bottom: 18px;
    }
    </style>

    <div class="custom-box">
        <h1 class="custom-title">⚡ SCHTOKS – A Product of Compliance In Sight ⚡</h1>
    </div>
    """,
    unsafe_allow_html=True
)
     
# CSV file path
WATCHLIST_FILE = "stocks.csv"
DATA_FILE = "stock_data.csv"

# Load stock symbols from CSV (with extra columns if exist)
def load_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        df = pd.read_csv(WATCHLIST_FILE)
        if "Starred" not in df.columns:
            df["Starred"] = 0
        return df
    else:
        return pd.DataFrame(columns=["Symbol", "Industry", "Buy Price", "Target Price", "Stop Loss", "Starred"])

# Save watchlist to CSV
def save_watchlist(df):
    df.to_csv(WATCHLIST_FILE, index=False)

# Function to detect candle type
def detect_candle(open_price, high, low, close):
    body = abs(close - open_price)
    upper_shadow = high - max(open_price, close)
    lower_shadow = min(open_price, close) - low

    if body == 0:
        return "Doji"
    elif body < (high - low) * 0.2:
        return "Doji-like"
    elif close > open_price:
        if upper_shadow < body * 0.3 and lower_shadow > body:
            return "Hammer-Bullish"
        elif body > (high - low) * 0.6:
            return "Super Bullish"
        else:
            return "Mid Bullish"
    else:
        if lower_shadow < body * 0.3 and upper_shadow > body:
            return "Inverted Hammer-Bearish"
        elif body > (high - low) * 0.6:
            return "Super Bearish"
        else:
            return "Mid Bearish"

# Format numbers nicely
def fmt(x):
    if pd.isna(x):
        return "-"
    return f"{x:,.2f}".rstrip("0").rstrip(".")

# Function to fetch OHLC + Current Price + Volume info

def fetch_stock_data(watchlist):
    results = []
    notifications = []  # store messages to send together
    for _, row in watchlist.iterrows():
        symbol = row["Symbol"]
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1y", interval="1d")
            if data.empty:
                continue

            latest = data.iloc[-1]
            current_price = ticker.fast_info.get("last_price", latest["Close"])

            buy_price = row.get("Buy Price", None)
            target_price = row.get("Target Price", None)
            stop_loss = row.get("Stop Loss", None)
            
            # Fetch company name (fallback to symbol if not found)
            info = ticker.info
            company_name = info.get("shortName", symbol)

            # Sector Info
            
            sector = ticker.info.get("sector", "-")
            industry = ticker.info.get("industry", "-")
            
            # EMA & DMA
            ema9_raw = data['Close'].ewm(span=9, adjust=False).mean().iloc[-1]
            ema21_raw = data['Close'].ewm(span=21, adjust=False).mean().iloc[-1]
            dma200_raw = data['Close'].rolling(window=200).mean().iloc[-1]

            # Ticker recommendation 
            recs = ticker.recommendations_summary
            if not recs.empty:
                latest_rec = recs.iloc[-1]  # last available recommendation summary
                strong_buy = latest_rec.get("strongBuy", 0)
                buy = latest_rec.get("buy", 0)
                hold = latest_rec.get("hold", 0)
                sell = latest_rec.get("sell", 0)
                strong_sell = latest_rec.get("strongSell", 0)
                analyst_view = f"Buy:{buy} Hold:{hold} Sell:{sell}"
            else:
                analyst_view = "-"
                
            # Daily % change from yfinance directly
            daily_pct_change = data["Close"].pct_change().iloc[-1] * 100 if len(data) > 1 else None

            # Volume classification
            avg20_volume = data["Volume"].tail(20).mean()
            avg50_volume = data["Volume"].tail(50).mean()
            today_volume = latest["Volume"]
            vol20_diff = ((today_volume - avg20_volume) / avg20_volume) * 100
            vol50_diff = ((today_volume - avg50_volume) / avg50_volume) * 100
            if vol20_diff > 20 and avg20_volume > avg50_volume:
                volume_status = f"Strong High Vm (+{vol20_diff:.1f}%, +{vol50_diff:.1f}% vs 50D)"
            elif vol20_diff > 20:
                volume_status = f"High Vm (+{vol20_diff:.1f}%)"
            elif vol20_diff < -20 and avg20_volume < avg50_volume:
                volume_status = f"Weak Low Vm ({vol20_diff:.1f}%, {vol50_diff:.1f}% vs 50D)"
            elif vol20_diff < -20:
                volume_status = f"Low Vm ({vol20_diff:.1f}%)"
            else:
                volume_status = f"Avg Vm ({vol20_diff:.1f}%)"

            # RSI Calculation
            rsi_series = ta.momentum.RSIIndicator(data["Close"], window=14).rsi()
            rsi_value = rsi_series.iloc[-1]
            rsi_status = "-"
            if rsi_value >= 80:
                rsi_status = f"Overbought ({fmt(rsi_value)})"
            elif rsi_value <= 20:
                rsi_status = f"Oversold ({fmt(rsi_value)})"
            elif 60 <= rsi_value < 80:
                rsi_status = f"Entering Overbought ({fmt(rsi_value)})"
            elif 20 < rsi_value <= 30:
                rsi_status = f"Entering Oversold ({fmt(rsi_value)})"
            else:
                rsi_status = f"Neutral ({fmt(rsi_value)})"

            # Calculate % changes
            pct_from_buy = ((current_price - buy_price) / buy_price * 100) if pd.notna(buy_price) and buy_price != 0 else None
            pct_from_target = ((current_price - target_price) / target_price * 100) if pd.notna(target_price) and target_price != 0 else None
            pct_from_stop = ((current_price - stop_loss) / stop_loss * 100) if pd.notna(stop_loss) and stop_loss != 0 else None

            # Target/Stop Loss status
            status = "-"
            if pd.notna(target_price) and target_price > 0:
                if current_price >= target_price:
                    status = "✅Target Hit"
                elif current_price >= target_price * 0.98:
                    status = "⚠️ Near Target"
            if pd.notna(stop_loss) and stop_loss > 0:
                if current_price <= stop_loss:
                    status = "❌ Stop Loss Hit"
                elif current_price <= stop_loss * 1.02:
                    status = "⚠️ Near Stop Loss"
                    
            # ----- Weighted Scoring System -----
            score = 0
            
            # RSI (20%)
            if rsi_value is not None:
                if rsi_value <= 30:
                    score += 20
                elif rsi_value >= 70:
                    score -= 20
            
            # EMA crossover (30%)
            if ema9_raw is not None and ema21_raw is not None:
                if ema9_raw > ema21_raw:
                    score += 30
                elif ema9_raw < ema21_raw:
                    score -= 30
                    
            # Price vs DMA200 (30%)
            if dma200_raw is not None:
                if current_price > dma200_raw:
                    score += 30
                else:
                    score -= 30
                
            # ----- Final Signal -----
            if score >= 60:
                final_signal = f"Strong Buy: {score}"
            elif score >= 30:
                final_signal = f"Buy: {score}"
            elif score <= -60:
                final_signal = f"Strong Sell: {score}"
            elif score <= -30:
                final_signal = f"Sell: {score}"
            else:
                final_signal = f"Hold: {score}"
           
            candle_type = detect_candle(latest["Open"], latest["High"], latest["Low"], latest["Close"])

            results.append({
                "Name": company_name,
                #"Open": fmt(latest["Open"]),
                #"High": fmt(latest["High"]),
                #"Low": fmt(latest["Low"]),
                #"Close": fmt(latest["Close"]),
                "Current Price": fmt(current_price),
                "Daily%": fmt(daily_pct_change) if daily_pct_change is not None else "-",
                "Buy Price": fmt(buy_price) if pd.notna(buy_price) else "-",
                "Target Price": fmt(target_price) if pd.notna(target_price) else "-",
                "Stop Loss": fmt(stop_loss) if pd.notna(stop_loss) else "-",
                "%FromBuy": fmt(pct_from_buy) if pct_from_buy is not None else "-",
                "%FromTgT": fmt(pct_from_target) if pct_from_target is not None else "-",
                "%FromSL": fmt(pct_from_stop) if pct_from_stop is not None else "-",
                "Volume Status": volume_status,
                "Candle Type": candle_type,
                "RSI": rsi_status,
                "Analyst View": analyst_view,
                "ema9": fmt(((current_price - ema9_raw) / ema9_raw) * 100) if ema9_raw else "-",
                "ema21": fmt(((current_price - ema21_raw) / ema21_raw) * 100) if ema21_raw else "-",
                "dma200": fmt(((current_price - dma200_raw) / dma200_raw) * 100) if dma200_raw else "-",
                "Target/SL Status": status,
                "Signal": final_signal,
                "Industry": industry,
                "Symbol": symbol,
            })
        except Exception:
            results.append({
                "Symbol": symbol,
                "Industry": row.get("Industry", "-"),
                "Current Price": "-",
                "Buy Price": row.get("Buy Price", "-"),
                "Target Price": row.get("Target Price", "-"),
                "Stop Loss": row.get("Stop Loss", "-"),
                "%FromBuy": "-",
                "%FromTgT": "-",
                "%FromSL": "-",
                "Daily%": "-",
                "Volume Status": "-",
                "Candle Type": "-",
                "RSI": "-",
                "Target/SL Status": "-",
                "analyst_view" : "-",
                "ema9": "-",
                "ema21": "-",
                "dma200": "-",
                "Signal": "-",
                "Name": "-",
            })
    # Send one combined message if there are alerts
    send_combined_notifications(notifications)
    
    df = pd.DataFrame(results)
    if not df.empty:
        df.to_csv(DATA_FILE, index=False)  # Save table data to CSV automatically
    return df

watchlist = load_watchlist()


# Add new stock to watchlist
if st.checkbox("➕ Show Add Stock Form"):
    with st.form("add_stock_form"):
        new_symbol = st.text_input("Stock Symbol (e.g., RELIANCE.NS)")
        new_industry = st.text_input("Industry")
        new_buy = st.number_input("Buy Price", value=0.0, format="%.2f")
        new_target = st.number_input("Target Price", value=0.0, format="%.2f")
        new_stop = st.number_input("Stop Loss", value=0.0, format="%.2f")
        add_btn = st.form_submit_button("Add to Watchlist")

        if add_btn and new_symbol:
            new_row = pd.DataFrame(
                [[new_symbol, new_industry, new_buy, new_target, new_stop, 0]],
                columns=["Symbol", "Industry", "Buy Price", "Target Price", "Stop Loss", "Starred"]
            )
            watchlist = pd.concat([watchlist, new_row], ignore_index=True)
            save_watchlist(watchlist)
            st.success(f"Added {new_symbol} to watchlist!")

col1, col2 = st.columns([1, 1])

# =========================
# Auto Refresh Settings
# =========================

with col1:
    st.subheader("⚙️ Refresh Settings")

# Let user choose refresh interval
    interval_options = {
        "1 Minute": 60_000,
        "2 Minutes": 120_000,
        "5 Minutes": 300_000,
        "10 Minutes": 600_000
    }

# Initialize session state if not set
    if "refresh_interval" not in st.session_state:
        st.session_state.refresh_interval = 300_000  # default 5 minutes

    selected_label = st.selectbox(
        "Select auto-refresh interval:",
        list(interval_options.keys()),
        index=list(interval_options.values()).index(st.session_state.refresh_interval)
    )

# Update session state
    st.session_state.refresh_interval = interval_options[selected_label]

# Refresh control
    market_open = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0).time()
    market_close = datetime.now().replace(hour=15, minute=31, second=0, microsecond=0).time()
    now = datetime.now().time()

    if market_open <= now <= market_close:
        refresh_count = st_autorefresh(interval=st.session_state.refresh_interval, key="price_refresh")
        last_refresh_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(
            f"⏳ **Auto-refresh every {selected_label} (Market hours)** &nbsp;&nbsp;&nbsp;   | ⏱️ **Last Refreshed: {last_refresh_time}**",
            unsafe_allow_html=True
        )
    else:
        last_refresh_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(
            f"⏸ **Auto-refresh paused (Outside market hours)** &nbsp;&nbsp;&nbsp;  |  ⏱️ **Last Refreshed: {last_refresh_time}**",
            unsafe_allow_html=True
        )
  
# --- Starred Stocks Management ---
with col2:
    st.subheader("⭐ Favorites")

# Search box for symbol
    search_symbol = st.text_input("Search stock to star/unstar", placeholder="e.g. RELIANCE.NS")

    if search_symbol:
        if search_symbol in watchlist["Symbol"].values:
            row_index = watchlist.index[watchlist["Symbol"] == search_symbol][0]
            is_starred = bool(watchlist.at[row_index, "Starred"])

            new_status = st.checkbox(f"⭐ Mark {search_symbol} as favorite", value=is_starred)

            if new_status != is_starred:
                watchlist.at[row_index, "Starred"] = int(new_status)
                save_watchlist(watchlist)
                st.success(f"{'Starred' if new_status else 'Unstarred'} {search_symbol}")
                st.rerun()
        else:
            st.warning("Symbol not found in watchlist.")

# Sort starred on top
watchlist = watchlist.sort_values(by="Starred", ascending=False).reset_index(drop=True)

# Fetch stock data
if not watchlist.empty:
    df = fetch_stock_data(watchlist)

    # Highlight gains/losses
    def colorize(val):
        try:
            v = float(val.replace(",", "")) if isinstance(val, str) and val not in ["-", None] else None
        except:
            v = None
        if v is None:
            return ''
        color = 'green' if v > 0 else 'red' if v < 0 else 'black'
        return f'color: {color}'

    st.dataframe(df.style.map(colorize, subset=["%FromBuy", "%FromTgT", "%FromSL", "Daily%", "ema9", "ema21", "dma200"]), use_container_width=True, 
    height=15 * 35 + 50  # ~20 rows visible
    )


    # Option to download CSV (table data)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="💾 Download Table Data as CSV",
        data=csv,
        file_name="stock_data.csv",
        mime="text/csv",
    )
else:
    st.info("No stocks in watchlist. Add some above!")
    
# Function to generate mini stock chart
@st.cache_data(ttl=600)
def stock_chart(symbol):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1y", interval="1d")  # use 1 year so 200DMA works
        if data.empty:
            return None

        # Calculate EMA9, EMA21, DMA200
        data["EMA9"] = data["Close"].ewm(span=9).mean()
        data["EMA21"] = data["Close"].ewm(span=21).mean()
        data["DMA200"] = data["Close"].rolling(window=200).mean()

        fig = go.Figure()

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"],
            name="Price", increasing_line_color="green", decreasing_line_color="red"
        ))

        # EMA overlays
        fig.add_trace(go.Scatter(x=data.index, y=data["EMA9"],
                                 line=dict(color="blue", width=1), name="EMA9"))
        fig.add_trace(go.Scatter(x=data.index, y=data["EMA21"],
                                 line=dict(color="orange", width=1), name="EMA21"))

        # DMA200 overlay
        fig.add_trace(go.Scatter(x=data.index, y=data["DMA200"],
                                 line=dict(color="purple", width=2, dash="dot"), name="DMA200"))

        # Layout tweaks (compact size)
        fig.update_layout(
            height=300, width=500,
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis_rangeslider_visible=False,
            template="plotly_white"
        )
        return fig
    except Exception as e:
        st.error(f"Error fetching chart for {symbol}: {e}")
        return None

# =========================
# Show Charts Section (Expand/Collapse + Lazy Load)
# =========================
st.subheader("📊 Stock Charts")

if not watchlist.empty:
    # Build mapping Symbol → Name
    name_map = dict(zip(df["Symbol"], df["Name"]))

    # Initialize session state for expand/collapse toggle
    if "expand_charts" not in st.session_state:
        st.session_state.expand_charts = False

    # Button to toggle expand/collapse
    if st.button("🔄 Expand/Collapse All Charts"):
        st.session_state.expand_charts = not st.session_state.expand_charts

    for sym in watchlist["Symbol"].unique():
        company_name = name_map.get(sym, sym)

        # Expander is controlled by global expand/collapse state
        with st.expander(f"📈 {company_name} Chart", expanded=st.session_state.expand_charts):
            # Lazy load: only generate chart if expander is visible/open
            if st.session_state.expand_charts or st.session_state.get(f"open_{sym}", False):
                fig = stock_chart(sym)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"No data available for {company_name}")

