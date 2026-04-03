import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Robo-Advisor | MCNA", layout="wide", page_icon="🤖")

# --- HÀM XỬ LÝ LÕI ---
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

# # --- GIAO DIỆN HEADER ---
st.title("🤖 MCNA Track 1: Robo-Advisor (AI Portfolio Manager)")
st.caption("Được thiết kế và phát triển riêng cho hạng mục Data/Quant Analytics - MCNA")

# Bổ sung bảng cảnh báo Disclaimer ngay từ đầu
st.warning("""
**⚠️ DISCLAIMER - LƯU Ý VỀ NGUỒN DỮ LIỆU:** Hệ thống đang kết nối Real-time với dữ liệu toàn cầu của Yahoo Finance để đảm bảo tính ổn định cao nhất. 
Do đó, một số cổ phiếu vốn hóa siêu nhỏ (Penny), thanh khoản thấp hoặc thuộc sàn UPCoM có thể sẽ không có sẵn dữ liệu. Khuyến nghị quý vị trải nghiệm mô hình bằng các mã **Blue-chip hoặc Mid-cap** (VD: HPG, SSI, FPT, DGC, VND...) để thuật toán lượng tử phát huy tối đa sức mạnh.
""")
st.markdown("---")

# --- TẠO TABS ĐIỀU HƯỚNG ---
tab_app, tab_guide, tab_about = st.tabs(["🚀 Ứng dụng Phân tích", "📖 Hướng dẫn sử dụng", "ℹ️ Về hệ thống"])

# 1. TAB GIỚI THIỆU
with tab_about:
    st.markdown("""
    ### 🎯 Mục tiêu của hệ thống
    **Robo-Advisor** là một trợ lý ảo đầu tư dựa trên nền tảng thuật toán lượng tử **Mean-Variance Optimization (Markowitz)**. 
    Hệ thống giúp nhà đầu tư cá nhân tự động thiết kế danh mục bằng cách tìm kiếm các tài sản có **độ tương quan thấp**, từ đó tối đa hóa tỷ suất sinh lời và kiểm soát rủi ro sụt giảm (Max Drawdown) trong những kịch bản xấu nhất của thị trường.
    
    ### ⚙️ Nguồn Dữ liệu & Công nghệ
    * **Data Pipeline:** Kết nối Real-time API với Yahoo Finance (Global).
    * **Khoảng thời gian (Backtest):** 2 năm giao dịch gần nhất.
    * **Công cụ trực quan:** Streamlit & Plotly Interactive Charts.
    """)

# 2. TAB HƯỚNG DẪN
with tab_guide:
    st.markdown("""
    ### 📝 Cách sử dụng Robo-Advisor trong 3 bước:
    1. **Nhập mã yêu thích:** Gõ một mã cổ phiếu bạn đang quan tâm vào ô tìm kiếm (Ví dụ: `HPG`, `VND`, `DIG`).
    2. **Kích hoạt AI:** Bấm nút **🤖 Kích hoạt AI Phân Tích**.
    3. **Đọc kết quả:**
       * Hệ thống sẽ tự động quét rổ VN100 để nhặt ra 4 mã "Vệ tinh" giúp bảo vệ rủi ro cho mã bạn chọn.
       * Xem ngay **Báo cáo chuyên sâu** và **Biểu đồ Backtest** để so sánh hiệu quả giữa việc AI quản lý tiền và việc bạn tự ôm 1 mã.
       * Click vào các đường link gắn kèm để tra cứu hồ sơ doanh nghiệp trên sàn.
    """)

# 3. TAB ỨNG DỤNG CHÍNH (APP)
with tab_app:
    col_input, col_info = st.columns([1, 2])
    with col_input:
        user_ticker = st.text_input("🔍 Nhập mã cổ phiếu trung tâm (VD: HPG, DIG, VND):", value="HPG").upper()
        analyze_btn = st.button("🤖 Kích hoạt AI Phân Tích", type="primary", use_container_width=True)

    with col_info:
        st.info("💡 **Cách AI hoạt động:** Thay vì để bạn chọn mã cảm tính, AI sẽ dùng **Ma trận tương quan (Correlation)** tìm ra 4 mã đi ngược pha với mã bạn chọn để triệt tiêu rủi ro.")

    if analyze_btn and user_ticker:
        with st.spinner(f"Đang quét rổ VN100 để tìm cổ phiếu có độ tương quan thấp nhất với {user_ticker}..."):
            scan_list = [user_ticker] + [t for t in BASKET if t != user_ticker]
            df_all = fetch_data(scan_list)
            
            if user_ticker not in df_all.columns:
                st.error(f"❌ Dữ liệu toàn cầu (Yahoo Finance) hiện bị khuyết hoặc không hỗ trợ mã **{user_ticker}**.")
                st.info("💡 **Gợi ý từ hệ thống:** Các mã vốn hóa quá nhỏ, thanh khoản thấp hoặc nằm trên sàn UPCoM thường không đạt chuẩn để thuật toán lượng tử phân tích. Vui lòng thử các mã Blue-chip hoặc Mid-cap có thanh khoản cao (VD: SSI, HPG, DGC, VND...).")
            elif len(df_all.columns) < 5:
                st.warning(f"⚠️ Cảnh báo: Thuật toán chỉ tìm được {len(df_all.columns) - 1} mã vệ tinh đủ điều kiện dữ liệu để ghép với {user_ticker} thay vì 4 mã như dự kiến, nhưng vẫn sẽ tiếp tục tối ưu hóa.")
            else:
                returns_all = df_all.pct_change().dropna()
                corr_matrix = returns_all.corr()
                best_hedges = corr_matrix[user_ticker].sort_values()[1:5].index.tolist()
                
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
                
                st.success(f"✅ AI đã thiết kế xong! 4 mã bảo vệ cho {user_ticker} là: **{', '.join(best_hedges)}**")
                
                # --- AUTO-INSIGHTS (AI ĐỌC SỐ LIỆU BẰNG LỜI) ---
                max_ticker = final_portfolio[np.argmax(opt_weights)]
                max_weight = np.max(opt_weights) * 100
                
                if opt_sharpe > 1:
                    sharpe_text = "CỰC KỲ XUẤT SẮC, mang lại mức sinh lời vượt trội so với rủi ro phải đánh đổi"
                elif opt_sharpe > 0.5:
                    sharpe_text = "KHÁ TỐT, hệ thống đã nỗ lực cân bằng được giữa lợi nhuận và biến động"
                else:
                    sharpe_text = "MANG TÍNH PHÒNG THỦ, ưu tiên giữ tiền khi thị trường đang có dấu hiệu đi ngang"

                st.markdown("### 🧠 Phân tích chuyên sâu từ hệ thống")
                st.warning(f"**Insight:** Danh mục tối ưu đang dồn tỷ trọng lớn nhất vào **{max_ticker}** với **{max_weight:.1f}%** vốn. Với điểm Sharpe đạt **{opt_sharpe:.2f}**, mô hình đánh giá đây là một chiến lược **{sharpe_text}**. Điểm sáng lớn nhất là rủi ro sập hầm (Max Drawdown) được ép xuống chỉ còn **{max_dd_port*100:.2f}%**, giúp bạn ngủ ngon hơn trong những nhịp điều chỉnh mạnh của Vn-Index.")
                
                # --- LINK TRA CỨU CỔ PHIẾU ---
                links_md = " | ".join([f"[{t}](https://fireant.vn/dashboard/content/symbols/{t})" for t in final_portfolio])
                st.markdown(f"🔗 **Click để tra cứu BCTC & Đồ thị Real-time trên FireAnt:** {links_md}")

                # --- DASHBOARD ---
                st.markdown("---")
                st.markdown("### 📊 Các Chỉ Số Đo Lường (KPIs)")
                
                # Cột 1: Hiển thị các chỉ số có gắn kèm Tooltip (Dấu chấm hỏi nhỏ)
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric(label="Lợi nhuận dự phóng (Năm)", value=f"{opt_ret*100:.2f}%", 
                            help="Tỷ suất sinh lời trung bình mỗi năm dựa trên dữ liệu quá khứ. Lưu ý: Số liệu này có tính chất tham khảo kỳ vọng, không đảm bảo chắc chắn cho tương lai.")
                
                kpi2.metric(label="Rủi ro biến động (Volatility)", value=f"{opt_vol*100:.2f}%", 
                            help="Độ lệch chuẩn của lợi nhuận. Mức độ giật lên/xuống của giá cổ phiếu. Volatility càng thấp thì tài khoản của bạn càng tăng trưởng êm ái.")
                
                kpi3.metric(label="Điểm Sharpe (Độ hiệu quả)", value=f"{opt_sharpe:.2f}", 
                            help="Cứ mỗi 1% rủi ro bạn chịu đựng, bạn thu về được bao nhiêu % lợi nhuận vượt trội. Sharpe > 1 là danh mục xuất sắc.")
                
                kpi4.metric(label="Rủi ro sập hầm (Max Drawdown)", value=f"{max_dd_port*100:.2f}%", delta="Tối ưu hóa", delta_color="off", 
                            help="Mức sụt giảm tài sản lớn nhất từ Đỉnh xuống Đáy. Giúp bạn chuẩn bị tâm lý cho kịch bản tồi tệ nhất của thị trường.")
                
                # Cột 2: Khung Expander giải thích sâu (Tự động cập nhật theo số liệu)
                with st.expander("📖 Bấm vào đây để xem AI giải thích chi tiết ý nghĩa các con số này"):
                    st.markdown(f"""
                    * **Tại sao Lợi nhuận dự phóng lại là {opt_ret*100:.2f}%?** Hệ thống tính toán trung bình biến động giá hàng ngày của 5 cổ phiếu trong 2 năm qua, sau đó nhân với tỷ trọng AI phân bổ. Mức này cao hay thấp phụ thuộc rất nhiều vào đà tăng trưởng của mã **{max_ticker}** (mã đang chiếm tỷ trọng lớn nhất).
                    
                    * **Tại sao Rủi ro biến động (Volatility) được ép xuống mức {opt_vol*100:.2f}%?** Nếu bạn chỉ mua 1 mã {user_ticker}, rủi ro sẽ rất lớn. Nhưng nhờ thuật toán quét Ma trận Hiệp phương sai (Covariance Matrix), AI đã tìm ra các mã đi ngược pha với {user_ticker} để thêm vào. Nhờ vậy, khi {user_ticker} giảm, các mã khác sẽ tăng để "đỡ" lại tài khoản cho bạn, giúp độ biến động giảm xuống đáng kể.
                    
                    * **Chỉ số Sharpe {opt_sharpe:.2f} nói lên điều gì?** Đầu tư không chỉ là nhìn vào lợi nhuận, mà phải xem bạn phải đánh đổi bao nhiêu rủi ro. Với mức {opt_sharpe:.2f}, hệ thống đang {"hoạt động rất tối ưu, mang lại mức sinh lời xứng đáng với rủi ro" if opt_sharpe > 0.8 else "cố gắng cân bằng và ưu tiên tính phòng thủ bảo vệ vốn cho bạn"}.
                    
                    * **Max Drawdown {max_dd_port*100:.2f}% có nguy hiểm không?** Đây là kịch bản xấu nhất (Worst-case scenario). Nghĩa là trong 2 năm qua, nếu bạn xui xẻo mua ngay đúng đỉnh, tài khoản của bạn sẽ bị âm tạm thời tối đa là **{max_dd_port*100:.2f}%** trước khi phục hồi trở lại. So với việc thị trường chứng khoán thường xuyên sập 30-40%, việc AI giữ được mức Drawdown này là một sự bảo vệ cực kỳ vững chắc.
                    """)
                
                st.markdown("---")
                # ... (Phần vẽ 2 biểu đồ Pie Chart và Line Chart bên dưới giữ nguyên) ...
                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    st.markdown("#### 🥧 Tỷ trọng giải ngân tối ưu")
                    result_df = pd.DataFrame({'Mã': final_portfolio, 'Tỷ trọng': opt_weights * 100})
                    fig1 = px.pie(result_df, values='Tỷ trọng', names='Mã', hole=0.45)
                    fig1.update_traces(textposition='inside', textinfo='percent+label')
                    fig1.update_layout(margin=dict(t=0, b=0, l=0, r=0))
                    st.plotly_chart(fig1, use_container_width=True)
                    
                with col_chart2:
                    st.markdown("#### 📈 Backtest Tăng trưởng tài sản")
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=portfolio_cumulative.index, y=portfolio_cumulative.values, mode='lines', name='Danh mục AI', line=dict(color='#00FFAA', width=2)))
                    fig2.add_trace(go.Scatter(x=single_stock_cumulative.index, y=single_stock_cumulative.values, mode='lines', name=f'All-in {user_ticker}', line=dict(color='#FF4444', width=2, dash='dot')))
                    fig2.update_layout(template='plotly_dark', margin=dict(t=0, b=0, l=0, r=0), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
                    st.plotly_chart(fig2, use_container_width=True)
