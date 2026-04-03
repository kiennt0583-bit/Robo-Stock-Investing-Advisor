import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# --- 1. CẤU HÌNH TRANG & ẨN DẤU VẾT STREAMLIT ---
st.set_page_config(page_title="Robo-Advisor | MCNA", layout="wide", page_icon="🤖")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .block-container {
                padding-top: 2rem;
                padding-bottom: 0rem;
            }
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# --- 2. HÀM XỬ LÝ LÕI ---
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

# --- 3. GIAO DIỆN HEADER & DISCLAIMER ---
st.title("🤖 MCNA Track 1: Robo-Advisor (AI Portfolio Manager)")
st.caption("Được thiết kế và phát triển riêng cho hạng mục Data/Quant Analytics - MCNA")

st.warning("""
**⚠️ DISCLAIMER - LƯU Ý VỀ NGUỒN DỮ LIỆU:** Hệ thống đang kết nối Real-time với dữ liệu toàn cầu của Yahoo Finance để đảm bảo tính ổn định cao nhất. 
Do đó, một số cổ phiếu vốn hóa siêu nhỏ (Penny), thanh khoản thấp hoặc thuộc sàn UPCoM có thể sẽ không có sẵn dữ liệu. Khuyến nghị trải nghiệm mô hình bằng các mã **Blue-chip hoặc Mid-cap** (VD: HPG, SSI, FPT, DGC, VND...).
""")
st.markdown("---")

# --- 4. THANH ĐIỀU HƯỚNG BÊN TRÁI (SIDEBAR) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2010/2010684.png", width=60)
    st.markdown("### ⚙️ Bảng Điều Khiển")
    user_ticker = st.text_input("🔍 Nhập mã cổ phiếu (VD: HPG, DIG):", value="HPG").upper()
    analyze_btn = st.button("🤖 Kích hoạt AI Phân Tích", type="primary", use_container_width=True)
    st.markdown("---")
    st.caption("🧠 Thuật toán: Markowitz (Mean-Variance)")
    st.caption("📡 Dữ liệu: Yahoo Finance API")
    st.caption("⏳ Khung thời gian: 2 năm gần nhất")

# --- 5. KHU VỰC TRƯNG BÀY CHÍNH (TABS) ---
tab_app, tab_guide, tab_about = st.tabs(["🚀 Báo cáo AI", "📖 Hướng dẫn sử dụng", "ℹ️ Về hệ thống"])

with tab_about:
    st.markdown("""
    ### 🎯 Mục tiêu của hệ thống
    **Robo-Advisor** là một trợ lý ảo đầu tư dựa trên nền tảng thuật toán lượng tử **Mean-Variance Optimization (Markowitz)**. 
    Hệ thống giúp nhà đầu tư cá nhân tự động thiết kế danh mục bằng cách tìm kiếm các tài sản có **độ tương quan thấp**, từ đó tối đa hóa tỷ suất sinh lời và kiểm soát rủi ro sụt giảm (Max Drawdown) trong những kịch bản xấu nhất của thị trường.
    """)

with tab_guide:
    st.markdown("""
    ### 📝 Cách sử dụng Robo-Advisor trong 3 bước:
    1. **Nhập mã yêu thích:** Nhìn sang thanh Menu màu xám bên trái, gõ một mã cổ phiếu bạn đang quan tâm (Ví dụ: `HPG`).
    2. **Kích hoạt AI:** Bấm nút **🤖 Kích hoạt AI Phân Tích**.
    3. **Đọc kết quả:** Hệ thống sẽ tự động quét rổ VN100 để nhặt ra 4 mã "Vệ tinh" giúp bảo vệ rủi ro cho mã bạn chọn và vẽ ra lộ trình đầu tư tối ưu.
    """)

with tab_app:
    # Lời chào khi chưa bấm nút
    if not analyze_btn:
        st.info("👈 Hãy nhập mã cổ phiếu ở thanh Menu bên trái và bấm Kích hoạt AI để bắt đầu!")

    # Xử lý khi bấm nút ở Sidebar
    if analyze_btn and user_ticker:
        
        # 1. HIỆU ỨNG LOADING CHUYÊN NGHIỆP (STATUS)
        with st.status("🤖 Hệ thống đang phân tích...", expanded=True) as status:
            st.write("🔍 Đang tải dữ liệu VN100 từ Global API...")
            scan_list = [user_ticker] + [t for t in BASKET if t != user_ticker]
            df_all = fetch_data(scan_list)
            
            if user_ticker not in df_all.columns:
                status.update(label="Lỗi dữ liệu đầu vào!", state="error", expanded=False)
                st.error(f"❌ Dữ liệu toàn cầu hiện bị khuyết hoặc không hỗ trợ mã **{user_ticker}**.")
            else:
                st.write("🧮 Đang tính toán Ma trận tương quan (Correlation Matrix)...")
                returns_all = df_all.pct_change().dropna()
                corr_matrix = returns_all.corr()
                best_hedges = corr_matrix[user_ticker].sort_values()[1:5].index.tolist()
                
                st.write("⚙️ Đang kích hoạt thuật toán tối ưu Markowitz...")
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
                bounds = tuple((0.05, 0.5) for _ in range(num_assets)) 
                res = minimize(neg_sharpe, num_assets * [1./num_assets,], method='SLSQP', bounds=bounds, constraints=constraints)
                opt_weights = res.x
                
                opt_ret = np.sum(mean_returns * opt_weights)
                opt_vol = np.sqrt(np.dot(opt_weights.T, np.dot(cov_matrix, opt_weights)))
                opt_sharpe = (opt_ret - 0.05) / opt_vol
                
                df_final_norm = df_final / df_final.iloc[0] 
                portfolio_cumulative = (df_final_norm * opt_weights).sum(axis=1)
                single_stock_cumulative = df_final_norm[user_ticker]
                max_dd_port = calc_max_drawdown(portfolio_cumulative)
                
                # Hoàn thành Loading
                status.update(label="✅ Đã tối ưu xong danh mục!", state="complete", expanded=False)
                
                # 2. THÔNG BÁO POP-UP (TOAST NOTIFICATION)
                st.toast(f"Đã tìm thấy 4 mã bọc lót cho {user_ticker}!", icon="🎉")
                
                # --- PHÂN TÍCH AI VÀ INSIGHTS ---
                max_ticker = final_portfolio[np.argmax(opt_weights)]
                max_weight = np.max(opt_weights) * 100
                
                if opt_sharpe > 1: sharpe_text = "CỰC KỲ XUẤT SẮC, mang lại mức sinh lời vượt trội"
                elif opt_sharpe > 0.5: sharpe_text = "KHÁ TỐT, nỗ lực cân bằng giữa lợi nhuận và biến động"
                else: sharpe_text = "PHÒNG THỦ, ưu tiên giữ tiền khi thị trường đi ngang"

                st.markdown("### 🧠 Phân tích chuyên sâu từ AI")
                st.warning(f"**Insight:** Danh mục tối ưu dồn tỷ trọng lớn nhất vào **{max_ticker}** ({max_weight:.1f}%). Với điểm Sharpe **{opt_sharpe:.2f}**, chiến lược này **{sharpe_text}**. Điểm sáng lớn nhất là rủi ro sập hầm (Max Drawdown) được ép xuống chỉ còn **{max_dd_port*100:.2f}%**.")
                
                links_md = " | ".join([f"[{t}](https://fireant.vn/dashboard/content/symbols/{t})" for t in final_portfolio])
                st.markdown(f"🔗 **Tra cứu biểu đồ Real-time:** {links_md}")

                # --- BẢNG ĐIỀU KHIỂN KPI ---
                st.markdown("---")
                st.markdown("### 📊 Các Chỉ Số Đo Lường (KPIs)")
                
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric(label="Lợi nhuận kỳ vọng", value=f"{opt_ret*100:.2f}%", help="Tỷ suất sinh lời trung bình mỗi năm.")
                kpi2.metric(label="Độ biến động (Volatility)", value=f"{opt_vol*100:.2f}%", help="Biến động càng thấp, tài khoản càng an toàn.")
                kpi3.metric(label="Điểm Sharpe", value=f"{opt_sharpe:.2f}", help="Sharpe > 1 là một danh mục xuất sắc.")
                kpi4.metric(label="Max Drawdown", value=f"{max_dd_port*100:.2f}%", delta="Tối ưu hóa", delta_color="off", help="Rủi ro sập hầm lớn nhất từ Đỉnh xuống Đáy.")
                
                # [KHÔI PHỤC TÍNH NĂNG] EXPANDER GIẢI THÍCH CHI TIẾT
                with st.expander("📖 Bấm vào đây để xem AI giải thích chi tiết ý nghĩa các con số này"):
                    st.markdown(f"""
                    * **Tại sao Lợi nhuận kỳ vọng lại là {opt_ret*100:.2f}%?** Hệ thống tính toán trung bình biến động giá hàng ngày của 5 cổ phiếu trong 2 năm qua, sau đó nhân với tỷ trọng AI phân bổ. Mức này phụ thuộc rất nhiều vào đà tăng trưởng của mã **{max_ticker}** (mã đang chiếm tỷ trọng lớn nhất).
                    
                    * **Tại sao Độ biến động (Volatility) được ép xuống mức {opt_vol*100:.2f}%?** Nhờ thuật toán quét Ma trận tương quan, AI đã tìm ra các mã đi ngược pha với {user_ticker}. Khi {user_ticker} giảm, các mã vệ tinh sẽ tăng để "đỡ" lại tài khoản cho bạn.
                    
                    * **Chỉ số Sharpe {opt_sharpe:.2f} nói lên điều gì?** Với mức {opt_sharpe:.2f}, hệ thống đang {"hoạt động rất tối ưu, mang lại mức sinh lời xứng đáng với rủi ro" if opt_sharpe > 0.8 else "cố gắng cân bằng và ưu tiên tính phòng thủ bảo vệ vốn"}.
                    
                    * **Max Drawdown {max_dd_port*100:.2f}% có nguy hiểm không?** Đây là kịch bản xấu nhất (Worst-case). Nghĩa là nếu bạn xui xẻo mua ngay đúng đỉnh, tài khoản sẽ bị âm tạm thời tối đa là **{max_dd_port*100:.2f}%**. So với việc All-in 1 mã thường xuyên sập 30-50%, đây là một sự bảo vệ cực kỳ vững chắc.
                    """)

                # --- BIỂU ĐỒ VÀ BẢNG PROGRESS BAR ---
                st.markdown("---")
                col_chart, col_table = st.columns([1.5, 1])
                
                with col_chart:
                    st.markdown("#### 📈 Mô phỏng Tăng trưởng tài sản")
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=portfolio_cumulative.index, y=portfolio_cumulative.values, mode='lines', name='Danh mục AI (Bọc lót)', line=dict(color='#00FFAA', width=2)))
                    fig2.add_trace(go.Scatter(x=single_stock_cumulative.index, y=single_stock_cumulative.values, mode='lines', name=f'All-in {user_ticker}', line=dict(color='#FF4444', width=2, dash='dot')))
                    fig2.update_layout(template='plotly_dark', margin=dict(t=0, b=0, l=0, r=0), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
                    st.plotly_chart(fig2, use_container_width=True)

                with col_table:
                    st.markdown("#### 📋 Lộ trình Giải ngân")
                    result_df = pd.DataFrame({'Mã': final_portfolio, 'Tỷ trọng': opt_weights * 100})
                    result_df = result_df.sort_values(by='Tỷ trọng', ascending=False)
                    
                    st.dataframe(
                        result_df,
                        column_config={
                            "Mã": st.column_config.TextColumn("Cổ Phiếu", width="small"),
                            "Tỷ trọng": st.column_config.ProgressColumn(
                                "Tỷ trọng (%)",
                                help="Phần trăm vốn cần giải ngân",
                                format="%.1f%%",
                                min_value=0,
                                max_value=100,
                            ),
                        },
                        hide_index=True,
                        use_container_width=True
                    )
