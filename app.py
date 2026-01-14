import streamlit as st
import yfinance as yf
import pandas as pd
import feedparser
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# --- APP CONFIG ---
st.set_page_config(page_title="AlphaTerminal Pro", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stMetric { background-color: #161a25; border: 1px solid #2a2e39; border-radius: 8px; padding: 15px; }
    [data-testid="stMetricValue"] { font-size: 24px; color: #d1d4dc; }
    .main { background-color: #0c0d10; }
    </style>
    """, unsafe_allow_html=True)

# --- CORE ENGINE: DATA CLEANING ---
def clean_df(df):
    if df is None or df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip() for c in df.columns]
    return df

# --- AUTHENTICATION ---
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if not st.session_state.logged_in:
    st.title("🛡️ AlphaTerminal Secure Login")
    u, p = st.text_input("User"), st.text_input("Pass", type="password")
    if st.button("Access"):
        if u == "admin" and p == "mypass123":
            st.session_state.logged_in = True
            st.rerun()
    st.stop()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("💎 AlphaTerminal")
page = st.sidebar.radio("Navigation", ["📈 Dashboard", "🤖 AI Forecast", "📊 Stock Comparison"])
st.sidebar.markdown("### 🧑‍💻 About the Developer")
st.sidebar.info(
    """
**Mohammed Muzzammil S**  
B.Tech IT • BSACIST  

**Skills:**  
- Python  
- Machine Learning  
- Stock Market Analytics  
- Data Visualization (Plotly)  
- Web Apps (Streamlit)

**Project:**  
AlphaTerminal Pro – A professional-grade market analysis dashboard.
    """)


# --- PAGE 1: DASHBOARD ---
if page == "📈 Dashboard":
    st.title("Live Market Dashboard")
    
    # Live Indices
    idx_cols = st.columns(3)
    indices = {"NIFTY 50": "^NSEI", "SENSEX": "^BSESN", "BANK NIFTY": "^NSEBANK"}
    for i, (name, sym) in enumerate(indices.items()):
        try:
            d = yf.download(sym, period="1d", interval="1m", progress=False)
            d = clean_df(d)
            curr = d['Close'].iloc[-1]
            prev = d['Open'].iloc[0]
            pct = ((curr - prev) / prev) * 100
            idx_cols[i].metric(name, f"₹{curr:,.2f}", f"{pct:+.2f}%")
        except: pass

    st.divider()

    col_chart, col_news = st.columns([2, 1])
    with col_chart:
        ticker = st.selectbox("Symbol", ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"])
        tf = st.radio("Range", ["1D", "1W", "1M", "1Y", "5Y"], horizontal=True)
        tf_map = {"1D":("1d","1m"), "1W":("5d","5m"), "1M":("1mo","1h"), "1Y":("1y","1d"), "5Y":("5y","1wk")}
        p, inter = tf_map[tf]
        
        hist = yf.download(ticker, period=p, interval=inter, progress=False)
        hist = clean_df(hist)
        if hist is not None:
            fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'])])
            fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

    with col_news:
        st.subheader("📰 Market Intel (Live News)")

        try:
            feed_url = "https://news.google.com/rss/search?q=Indian+stock+market&hl=en-IN&gl=IN&ceid=IN:en"
            rss = feedparser.parse(feed_url)

            if len(rss.entries) == 0:
                st.warning("No news found right now.")
            else:
                for entry in rss.entries[:8]:
                    st.markdown(f"**{entry.source.title}**")
                    st.markdown(f"[{entry.title}]({entry.link})")
                    st.caption(entry.published)
                    st.divider()

        except Exception as e:
            st.error("News service temporarily unavailable.")

# --- PAGE 2: AI FORECAST ---
elif page == "🤖 AI Forecast":
    st.title("🤖 ML Price Prediction (Linear Regression)")
    target = st.text_input("Enter Ticker", "RELIANCE.NS")
    days_to_predict = st.slider("Prediction Days", 5, 30, 15)

    if st.button("Train & Forecast"):
        df = yf.download(target, period="2y", progress=False)
        df = clean_df(df).dropna()
        
        # Training Data
        df['Day_Index'] = np.arange(len(df))
        X = df[['Day_Index']]
        y = df['Close']
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Create Future Dates
        last_idx = df['Day_Index'].iloc[-1]
        future_indices = np.array(range(last_idx + 1, last_idx + days_to_predict + 1)).reshape(-1, 1)
        predictions = model.predict(future_indices)
        
        # Plotting Actual vs Prediction
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Historical Price", line=dict(color="#26a69a")))
        
        # Future dates for plotting
        last_date = df.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]
        
        fig.add_trace(go.Scatter(x=future_dates, y=predictions, name="AI Prediction", line=dict(dash='dash', color='#ef5350')))
        fig.update_layout(title=f"{target} {days_to_predict}-Day Forecast", template="plotly_dark", height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.success(f"Estimated Price in {days_to_predict} days: ₹{predictions[-1]:,.2f}")

# --- PAGE 3: STOCK COMPARISON (RETURNS WISE) ---
elif page == "📊 Stock Comparison":
    st.title("📊 Capital Growth Comparison (₹100 Investment)")
    stocks_to_compare = st.text_input("Tickers (comma separated)", "TCS.NS, RELIANCE.NS, INFY.NS")
    duration = st.selectbox("Comparison Period", ["1y", "2y", "5y"])
    
    if st.button("Compare Performance"):
        tickers = [t.strip().upper() for t in stocks_to_compare.split(",")]
        data = yf.download(tickers, period=duration, progress=False)
        data = clean_df(data)

        # FIX DUPLICATE CLOSE COLUMNS
        if len(tickers) == 1:
            close_prices = pd.DataFrame({tickers[0]: data["Close"]})
        else:
            close_prices = data["Close"].copy()
            close_prices.columns = tickers

        # CAPITAL GROWTH (₹100 invested)
        capital_df = close_prices.apply(lambda x: (x / x.iloc[0]) * 100)

        # Plotting
        fig = go.Figure()
        for stock in capital_df.columns:
            fig.add_trace(go.Scatter(x=capital_df.index, y=capital_df[stock], name=stock))

        fig.update_layout(
            title=f"Capital Growth (₹100 Invested) - Last {duration}",
            yaxis_title="Portfolio Value (₹)",
            template="plotly_dark",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info("Shows how ₹100 invested in each stock has grown over time.")
