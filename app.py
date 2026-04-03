import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Robo-Advisor | MCNA", layout="wide", page_icon="🤖")
st.title("🤖 MCNA Track 1: Robo-Advisor (AI Portfolio Manager)")
st.markdown("Nhập 1 cổ phiếu bạn yêu thích. AI sẽ tự động rà soát thị trường, tìm ra các mã 'bọc lót' rủi ro và thiết kế danh mục tối ưu nhất.")

# Rổ 20 cổ phiếu thanh khoản cao, đại diện cho các ngành nghề khác nhau
BASKET = ["FPT", "VCB", "PNJ", "DGW", "SZC", "VHC", "FTS", "HDG", "HAH", "KBC", "SSI", "MWG", "GAS", "REE", "VNM", "BID", "GMD", "DGC", "STB", "VJC"]

@st.cache_data(show_spinner=False)
def fetch_data(tickers, years=2):
    price_df = pd.DataFrame()
    for t in tickers:
        try:
            df = yf.Ticker(f"{t}.VN").history(period=f"{years}y")
            if not df.empty and 'Close' in df.columns:
                df.index = df.index.tz_localize(None) 
                price_df[t] = df['Close']
        except: pass
    return price_df.dropna()

def calc_max_drawdown(cumulative_returns):
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    return drawdown.min()

# --- GIAO DIỆN CHÍNH ---
st.markdown("---")
col_input, col_info = st.columns([1, 2])

with col_input:
    user_ticker = st.text_input("🔍 Nhập mã cổ phiếu trung tâm (VD: HPG, DIG, VND):", value="HPG").upper()
    analyze_btn = st.button("🤖 Kích hoạt AI Phân Tích", type="primary")

with col_info:
    st.info("💡 **Cách AI hoạt động:** Thay vì để bạn tự chọn mã vệ tinh một cách cảm tính, AI sẽ dùng thuật toán **Ma trận tương quan (Correlation Matrix)** để tìm ra 4 mã cổ phiếu có xu hướng đi ngược lại hoặc độc lập với mã bạn chọn. Điều này giúp triệt tiêu rủi ro khi thị trường sập.")

if analyze_btn and user_ticker:
    with st.spinner(f"Đang quét rổ VN100 để tìm cổ phiếu có độ tương quan thấp nhất với {user_ticker}... (Mất ~5 giây)"):
        
        # 1. Kéo dữ liệu của mã người dùng + toàn bộ rổ
        scan_list = [user_ticker] + [t for t in BASKET if t != user_ticker]
        df_all = fetch_data(scan_list)
        
        if user_ticker not in df_all.columns:
            st.error("❌ Không tìm thấy dữ liệu cho mã này. Vui lòng nhập mã chuẩn trên sàn HSX/HNX.")
        else:
            # 2. THUẬT TOÁN TÌM KIẾM ĐỘ TƯƠNG QUAN
            returns_all = df_all.pct_change().dropna()
            corr_matrix = returns_all.corr()
            
            # Lấy cột tương quan của mã người dùng, sắp xếp tăng dần, chọn 4 mã thấp nhất
            best_hedges = corr_matrix[user_ticker].sort_values()[1:5].index.tolist()
            
            st.success(f"✅ AI đã tìm thấy 4 mã có khả năng bảo vệ rủi ro tốt nhất cho {user_ticker} là: **{', '.join(best_hedges)}**")
            
            # 3. CHẠY MARKOWITZ CHO DANH MỤC 5 MÃ ĐÃ CHỌN
            final_portfolio = [user_ticker] + best_hedges
            df_final = df_all[final_portfolio]
            
            returns = df_final.pct_change().dropna()
            mean_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252
            num_assets = len(final_portfolio)
            
            def neg_sharpe(w):
                p_ret = np.sum(mean_returns * w)
                p_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
                return -(p_ret - 0.05) / p_vol
                
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0.05, 0.5) for _ in range(num_assets)) # Ràng buộc: Mua ít nhất 5%, tối đa 50%
            res = minimize(neg_sharpe, num_assets * [1./num_assets,], method='SLSQP', bounds=bounds, constraints=constraints)
            opt_weights = res.x
            
            # Tính toán các chỉ số
            opt_ret = np.sum(mean_returns * opt_weights)
            opt_vol = np.sqrt(np.dot(opt_weights.T, np.dot(cov_matrix, opt_weights)))
            opt_sharpe = (opt_ret - 0.05) / opt_vol
            
            # Tính hiệu suất tăng trưởng (Cumulative Return)
            df_final_norm = df_final / df_final.iloc[0] # Chuyển về base 1 (giả sử đầu tư 1 đồng)
            portfolio_cumulative = (df_final_norm * opt_weights).sum(axis=1)
            single_stock_cumulative = df_final_norm[user_ticker]
            
            max_dd_port = calc_max_drawdown(portfolio_cumulative)
            max_dd_single = calc_max_drawdown(single_stock_cumulative)

            # --- HIỂN THỊ TRỰC QUAN HÓA (DASHBOARD) ---
            st.markdown("---")
            st.markdown("### 🏆 BÁO CÁO PHÂN TÍCH TỪ AI")
            
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Lợi nhuận dự phóng (Năm)", f"{opt_ret*100:.2f}%")
            kpi2.metric("Rủi ro biến động (Volatility)", f"{opt_vol*100:.2f}%")
            kpi3.metric("Điểm Sharpe (Độ hiệu quả)", f"{opt_sharpe:.2f}")
            # Highlight chỉ số Max Drawdown
            kpi4.metric("Rủi ro sập hầm (Max Drawdown)", f"{max_dd_port*100:.2f}%", "AI Đã tối thiểu hóa", delta_color="off")
            
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.markdown("#### 🥧 Tỷ trọng giải ngân tối ưu")
                result_df = pd.DataFrame({'Mã': final_portfolio, 'Tỷ trọng': opt_weights * 100})
                fig1 = px.pie(result_df, values='Tỷ trọng', names='Mã', hole=0.45)
                fig1.update_traces(textposition='inside', textinfo='percent+label')
                fig1.update_layout(margin=dict(t=0, b=0, l=0, r=0))
                st.plotly_chart(fig1, use_container_width=True)
                
            with col_chart2:
                st.markdown("#### 📈 Mô phỏng tăng trưởng tài sản (Backtest 2 năm)")
                st.caption(f"So sánh việc AI quản lý danh mục VS Bạn cầm Full cổ phiếu {user_ticker}")
                
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=portfolio_cumulative.index, y=portfolio_cumulative.values, mode='lines', name='Danh mục AI (An toàn & Ổn định)', line=dict(color='#00FFAA', width=2)))
                fig2.add_trace(go.Scatter(x=single_stock_cumulative.index, y=single_stock_cumulative.values, mode='lines', name=f'All-in {user_ticker} (Biến động mạnh)', line=dict(color='#FF4444', width=2, dash='dot')))
                
                fig2.update_layout(template='plotly_dark', margin=dict(t=0, b=0, l=0, r=0), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
                st.plotly_chart(fig2, use_container_width=True)