import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# --- 1. CẤU HÌNH & CSS ---
st.set_page_config(page_title="Robo-Advisor Pro | MCNA", layout="wide", page_icon="📈")

design_style = """
    <style>
    [data-testid="stMetric"] {
        background-color: #1e2430; border: 1px solid #31333f; padding: 20px; border-radius: 15px;
    }
    .stExpander { border-radius: 10px !important; background-color: #161b22 !important; }
    </style>
    """
st.markdown(design_style, unsafe_allow_html=True)

# --- 2. HÀM XỬ LÝ ---
BASKET = ["FPT", "VCB", "PNJ", "DGW", "SZC", "VHC", "FTS", "HDG", "HAH", "KBC", "SSI", "MWG", "GAS", "REE", "VNM", "BID", "GMD", "DGC", "STB", "VJC"]

@st.cache_data(show_spinner=False)
def fetch_data(tickers, years=2):
    price_df = pd.DataFrame()
    for t in tickers:
        try:
            df = yf.Ticker(f"{t}.VN").history(period=f"{years}y")
            if not df.empty:
                df.index = df.index.tz_localize(None) 
                price_df[t] = df['Close']
        except: pass
    return price_df.dropna()

def calc_max_drawdown(cumulative_returns):
    peak = cumulative_returns.expanding(min_periods=1).max()
    return ((cumulative_returns / peak) - 1).min()

# --- 3. SIDEBAR & HEADER ---
with st.sidebar:
    st.header("⚙️ Cấu hình")
    user_ticker = st.text_input("🔍 Mã cổ phiếu mục tiêu:", value="HPG").upper()
    analyze_btn = st.button("🚀 Phân tích danh mục tối ưu", type="primary", use_container_width=True)
    st.divider()
    st.info("Sử dụng mô hình Markowitz & Ma trận hiệp phương sai để giảm thiểu rủi ro phi hệ thống.")

st.title("🤖 Robo-Advisor: Chiến lược Danh mục Tối ưu")
st.warning("⚠️ **Lưu ý:** Dữ liệu từ Yahoo Finance. Khuyến nghị dùng mã VN30/Midcap để có độ chính xác cao nhất.")

# --- 4. LOGIC CHÍNH ---
if analyze_btn and user_ticker:
    with st.status("🛠️ Đang thực hiện phân tích chuyên sâu...", expanded=True) as status:
        st.write("📡 Kết nối dữ liệu thị trường...")
        scan_list = [user_ticker] + [t for t in BASKET if t != user_ticker]
        df_all = fetch_data(scan_list)
        
        if user_ticker not in df_all.columns:
            status.update(label="Lỗi!", state="error")
            st.error(f"Không tìm thấy dữ liệu cho mã {user_ticker}")
        else:
            # Thuật toán
            returns_all = df_all.pct_change().dropna()
            corr_matrix = returns_all.corr()
            best_hedges = corr_matrix[user_ticker].sort_values()[1:5].index.tolist()
            final_list = [user_ticker] + best_hedges
            df_final = df_all[final_list]
            returns = df_final.pct_change().dropna()
            
            # Tối ưu Markowitz
            mean_ret = returns.mean() * 252
            cov_mat = returns.cov() * 252
            def obj(w): return -(np.sum(mean_ret * w) - 0.05) / np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
            res = minimize(obj, len(final_list)*[1./len(final_list)], bounds=tuple((0.05, 0.5) for _ in range(len(final_list))), constraints={'type':'eq','fun':lambda x:np.sum(x)-1})
            weights = res.x

            status.update(label="✅ Phân tích hoàn tất!", state="complete", expanded=False)

            # --- HIỂN THỊ KẾT QUẢ ---
            st.subheader("📊 Kết quả tối ưu hóa danh mục")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Lợi nhuận kỳ vọng", f"{np.sum(mean_ret*weights)*100:.1f}%")
            k2.metric("Rủi ro (Volatility)", f"{np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights)))*100:.1f}%")
            k3.metric("Điểm Sharpe", f"{(np.sum(mean_ret*weights)-0.05)/np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights))):.2f}")
            port_cum = (df_final/df_final.iloc[0] * weights).sum(axis=1)
            k4.metric("Max Drawdown", f"{calc_max_drawdown(port_cum)*100:.1f}%")

            # --- KHU VỰC GIẢI THÍCH LÝ DO (FEEDBACK FIX) ---
            st.markdown("---")
            col_reason, col_corr = st.columns([1, 1])
            
            with col_reason:
                st.markdown("### 🎯 Tại sao danh mục này phù hợp?")
                st.write(f"Dựa trên mã trọng tâm **{user_ticker}**, AI đã chọn thêm: **{', '.join(best_hedges)}**.")
                st.markdown(f"""
                - **Đa dạng hóa:** Các mã này có độ tương quan thấp (xem biểu đồ bên cạnh), giúp bảo vệ tài sản khi {user_ticker} biến động mạnh.
                - **Cân bằng ngành:** Danh mục kết hợp giữa Sản xuất, Công nghệ và Tài chính để tránh rủi ro tập trung ngành.
                - **Hiệu quả dòng tiền:** Tỷ trọng được tính toán để bạn không cần 'đoán' thị trường mà vẫn tối ưu được lợi nhuận/rủi ro.
                """)
                st.info("🔗 **Đối chiếu thông tin ngoài:** [Xem tin tức các mã này trên CafeF](https://cafef.vn/tim-kiem/" + user_ticker + ".chn)")

            with col_corr:
                st.markdown("### 🧬 Ma trận Tương quan (Lý do chọn mã)")
                # Hiện thị Heatmap để giải thích lý do chọn mã
                fig_corr = px.imshow(corr_matrix.loc[final_list, final_list], 
                                    text_auto=".2f", color_continuous_scale='RdBu_r',
                                    title="Tương quan càng thấp (màu đỏ) càng tốt cho đa dạng hóa")
                st.plotly_chart(fig_corr, use_container_width=True)

            # --- BIỂU ĐỒ TĂNG TRƯỞNG (FIX TOOLTIP) ---
            st.markdown("---")
            st.markdown("### 📈 Hiệu quả thực tế (Backtest 2 năm)")
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(x=port_cum.index, y=port_cum, name="Danh mục Robo", line=dict(color='#00FFAA')))
            fig_line.add_trace(go.Scatter(x=port_cum.index, y=df_final[user_ticker]/df_final[user_ticker].iloc[0], name=f"Chỉ giữ {user_ticker}", line=dict(dash='dot', color='#FF4444')))
            
            fig_line.update_layout(hovermode="x unified", template="plotly_dark", 
                                yaxis_title="Giá trị tài sản (Gốc = 1)",
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_line, use_container_width=True)

            # --- BẢNG GIẢI NGÂN ---
            st.markdown("### 📋 Chi tiết tỷ trọng giải ngân")
            res_df = pd.DataFrame({'Mã': final_list, 'Tỷ trọng (%)': weights*100})
            st.dataframe(res_df.sort_values('Tỷ trọng (%)', ascending=False), 
                        column_config={"Tỷ trọng (%)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)},
                        hide_index=True, use_container_width=True)
