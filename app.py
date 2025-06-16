import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import requests
from pathlib import Path

# Import our modules
from data_collector import BitcoinDataCollector
from predictor import BitcoinPredictor
from config import *

# Page config
st.set_page_config(
    page_title="Bitcoin Predictor - Bitget Data",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #f7931a;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">₿ Bitcoin Predictor - Powered by Bitget API</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Cài đặt")
    
    # Trading pair selection
    trading_pairs = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT"]
    selected_pair = st.sidebar.selectbox("Chọn cặp trading:", trading_pairs)
    
    # ✅ CẬP NHẬT: Chỉ hiển thị periods hoạt động
    timeframe_options = {
        "1 giờ": "1h",
        "4 giờ": "4h", 
        "6 giờ": "6h",
        "12 giờ": "12h",
        "1 tháng": "1M"
    }
    
    selected_timeframe_display = st.sidebar.selectbox(
        "Khung thời gian:", 
        list(timeframe_options.keys()),
        index=1  # Default to "4 giờ"
    )
    selected_timeframe = timeframe_options[selected_timeframe_display]
    
    # Data points
    data_points = st.sidebar.slider("Số điểm dữ liệu:", 50, 1000, 365)
    
    # Auto refresh
    auto_refresh = st.sidebar.checkbox("Tự động làm mới (30s)", False)
    
    if auto_refresh:
        st.rerun()
    
    # Main content
    try:
        # Initialize collector
        collector = BitcoinDataCollector()
        
        # Override symbol if needed
        if selected_pair != "BTCUSDT":
            collector.bitget_symbol = f"{selected_pair}_SPBL"
        
        # Show loading
        with st.spinner(f"🔄 Đang tải dữ liệu {selected_pair} ({selected_timeframe_display})..."):
            
            # Get data with specific timeframe
            df = get_data_with_timeframe(collector, selected_timeframe, data_points)
            
            if df is not None and len(df) > 10:
                st.success(f"✅ Đã tải thành công {len(df)} điểm dữ liệu!")
                
                # Display current price and stats
                display_price_metrics(df, selected_pair)
                
                # Create tabs
                tab1, tab2, tab3, tab4 = st.tabs(["📈 Biểu đồ", "🔮 Dự đoán", "📊 Phân tích", "📋 Dữ liệu"])
                
                with tab1:
                    display_price_chart(df, selected_pair, selected_timeframe_display)
                
                with tab2:
                    display_predictions(df, selected_pair)
                
                with tab3:
                    display_technical_analysis(df)
                
                with tab4:
                    display_raw_data(df)
                    
            else:
                st.error("❌ Không thể tải dữ liệu từ Bitget API. Vui lòng thử lại sau.")
                
                # Show fallback options
                st.info("🔄 Đang thử các nguồn dữ liệu khác...")
                
                # Try Binance as fallback
                df_binance = collector.get_binance_data(selected_pair, "1d", data_points)
                if df_binance is not None:
                    st.success("✅ Đã tải dữ liệu từ Binance!")
                    display_price_metrics(df_binance, selected_pair + " (Binance)")
                    display_price_chart(df_binance, selected_pair + " (Binance)", "Hàng ngày")
                else:
                    # Generate demo data
                    st.warning("⚠️ Sử dụng dữ liệu demo")
                    df_demo = collector.generate_dummy_data(data_points)
                    display_price_metrics(df_demo, selected_pair + " (Demo)")
                    display_price_chart(df_demo, selected_pair + " (Demo)", "Hàng ngày")
    
    except Exception as e:
        st.error(f"❌ Lỗi: {str(e)}")
        st.info("🔧 Đang sử dụng dữ liệu demo...")
        
        # Fallback to demo data
        collector = BitcoinDataCollector()
        df_demo = collector.generate_dummy_data(data_points)
        display_price_metrics(df_demo, selected_pair + " (Demo)")
        display_price_chart(df_demo, selected_pair + " (Demo)", "Demo")

def get_data_with_timeframe(collector, timeframe, data_points):
    """Lấy dữ liệu với timeframe cụ thể"""
    try:
        # Override the working periods to use selected timeframe
        collector.working_periods = [timeframe]
        
        # Get data
        df = collector.get_bitget_data(data_points)
        
        if df is not None:
            return df
        
        # Fallback to Binance
        interval_map = {
            '1h': '1h',
            '4h': '4h', 
            '6h': '6h',
            '12h': '12h',
            '1M': '1M'
        }
        
        binance_interval = interval_map.get(timeframe, '1d')
        return collector.get_binance_data(
            collector.bitget_symbol.replace('_SPBL', ''), 
            binance_interval, 
            data_points
        )
        
    except Exception as e:
        st.error(f"Lỗi khi lấy dữ liệu: {str(e)}")
        return None

def display_price_metrics(df, pair_name):
    """Hiển thị metrics giá"""
    if df is None or len(df) == 0:
        return
    
    current_price = df['close'].iloc[-1]
    prev_price = df['close'].iloc[-2] if len(df) > 1 else current_price
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
    
    # Price metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label=f"💰 Giá hiện tại ({pair_name})",
            value=f"${current_price:,.2f}",
            delta=f"{price_change_pct:+.2f}%"
        )
    
    with col2:
        high_24h = df['high'].tail(24).max() if len(df) >= 24 else df['high'].max()
        st.metric(
            label="📈 Cao nhất",
            value=f"${high_24h:,.2f}"
        )
    
    with col3:
        low_24h = df['low'].tail(24).min() if len(df) >= 24 else df['low'].min()
        st.metric(
            label="📉 Thấp nhất", 
            value=f"${low_24h:,.2f}"
        )
    
    with col4:
        volume_24h = df['volume'].tail(24).sum() if len(df) >= 24 else df['volume'].sum()
        st.metric(
            label="📊 Khối lượng",
            value=f"{volume_24h:,.0f}"
        )

def display_price_chart(df, pair_name, timeframe):
    """Hiển thị biểu đồ giá"""
    if df is None or len(df) == 0:
        st.error("Không có dữ liệu để hiển thị biểu đồ")
        return
    
    # Create candlestick chart
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f'{pair_name} - {timeframe}', 'Khối lượng'),
        row_width=[0.7, 0.3]
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=pair_name,
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        ),
        row=1, col=1
    )
    
    # Volume
    colors = ['#00ff88' if close >= open else '#ff4444' 
              for close, open in zip(df['close'], df['open'])]
    
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f"📈 Biểu đồ nến {pair_name}",
        xaxis_rangeslider_visible=False,
        height=600,
        showlegend=False,
        template="plotly_dark"
    )
    
    fig.update_xaxes(title_text="Thời gian", row=2, col=1)
    fig.update_yaxes(title_text="Giá ($)", row=1, col=1)
    fig.update_yaxes(title_text="Khối lượng", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

def display_predictions(df, pair_name):
    """Hiển thị dự đoán"""
    if df is None or len(df) < 30:
        st.warning("Cần ít nhất 30 điểm dữ liệu để dự đoán")
        return
    
    st.subheader("🔮 Dự đoán giá Bitcoin")
    
    try:
        # Initialize predictor
        predictor = BitcoinPredictor()
        
        with st.spinner("🤖 Đang phân tích và dự đoán..."):
            # Prepare data
            predictor.prepare_data(df)
            
            # Train model
            predictor.train_model()
            
            # Make predictions
            predictions = predictor.predict_next_days(days=7)
            
            if predictions is not None and len(predictions) > 0:
                # Display predictions
                st.success("✅ Dự đoán hoàn thành!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**📊 Dự đoán 7 ngày tới:**")
                    for i, price in enumerate(predictions):
                        date = (datetime.now() + timedelta(days=i+1)).strftime("%d/%m/%Y")
                        st.write(f"• {date}: ${price:,.2f}")
                
                with col2:
                    # Prediction chart
                    fig = go.Figure()
                    
                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=df.index[-30:],
                        y=df['close'][-30:],
                        mode='lines',
                        name='Lịch sử',
                        line=dict(color='#00ff88')
                    ))
                    
                    # Predictions
                    future_dates = pd.date_range(
                        start=df.index[-1] + timedelta(days=1),
                        periods=len(predictions),
                        freq='D'
                    )
                    
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=predictions,
                        mode='lines+markers',
                        name='Dự đoán',
                        line=dict(color='#ff4444', dash='dash')
                    ))
                    
                    fig.update_layout(
                        title="Dự đoán giá 7 ngày",
                        xaxis_title="Thời gian",
                        yaxis_title="Giá ($)",
                        template="plotly_dark",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Model performance
                st.write("**🎯 Hiệu suất mô hình:**")
                metrics = predictor.get_model_metrics()
                if metrics:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("MAE", f"{metrics.get('mae', 0):.2f}")
                    with col2:
                        st.metric("RMSE", f"{metrics.get('rmse', 0):.2f}")
                    with col3:
                        st.metric("R²", f"{metrics.get('r2', 0):.3f}")
            else:
                st.error("❌ Không thể tạo dự đoán")
                
    except Exception as e:
        st.error(f"❌ Lỗi khi dự đoán: {str(e)}")

def display_technical_analysis(df):
    """Hiển thị phân tích kỹ thuật"""
    if df is None or len(df) < 20:
        st.warning("Cần ít nhất 20 điểm dữ liệu để phân tích")
        return
    
    st.subheader("📊 Phân tích kỹ thuật")
    
    # Calculate technical indicators
    df_analysis = df.copy()
    
    # Moving averages
    df_analysis['MA7'] = df_analysis['close'].rolling(window=7).mean()
    df_analysis['MA25'] = df_analysis['close'].rolling(window=25).mean()
    
    # RSI
    delta = df_analysis['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_analysis['RSI'] = 100 - (100 / (1 + rs))
    
    # Display current values
    col1, col2, col3 = st.columns(3)
    
    with col1:
        current_rsi = df_analysis['RSI'].iloc[-1]
        rsi_signal = "Mua" if current_rsi < 30 else "Bán" if current_rsi > 70 else "Giữ"
        st.metric("RSI", f"{current_rsi:.1f}", rsi_signal)
    
    with col2:
        ma7 = df_analysis['MA7'].iloc[-1]
        ma25 = df_analysis['MA25'].iloc[-1]
        ma_signal = "Tăng" if ma7 > ma25 else "Giảm"
        st.metric("MA Signal", ma_signal)
    
    with col3:
        volatility = df_analysis['close'].pct_change().std() * 100
        st.metric("Độ biến động", f"{volatility:.2f}%")

def display_raw_data(df):
    """Hiển thị dữ liệu thô"""
    if df is None:
        st.error("Không có dữ liệu")
        return
    
    st.subheader("📋 Dữ liệu thô")
    
    # Display options
    col1, col2 = st.columns(2)
    with col1:
        show_rows = st.selectbox("Hiển thị:", [10, 25, 50, 100], index=1)
    with col2:
        if st.button("📥 Tải xuống CSV"):
            csv = df.to_csv()
            st.download_button(
                label="Tải file CSV",
                data=csv,
                file_name=f"bitcoin_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    # Show data
    st.dataframe(
        df.tail(show_rows).round(2),
        use_container_width=True
    )
    
    # Data summary
    st.write("**📈 Thống kê tóm tắt:**")
    st.dataframe(df.describe().round(2))

if __name__ == "__main__":
    main()
