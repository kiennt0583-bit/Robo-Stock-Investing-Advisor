import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from datetime import datetime, timedelta
from vnstock import Quote, register_user
register_user('vnstock_54f977a665bb05b32c1b9312821e7527')
import plotly.express as px
import plotly.graph_objects as go
import warnings
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

# --- 1. CẤU HÌNH & CSS ---
st.set_page_config(page_title="Robo-Advisor Pro | MCNA", layout="wide", page_icon="📈")

st.markdown("""
    <style>
    [data-testid="stMetric"] { background-color: #1e2430; border: 1px solid #31333f; padding: 20px; border-radius: 15px; }
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. HÀM XỬ LÝ ---
BASKET = ["FPT", "VCB", "PNJ", "DGW", "SZC", "VHC", "FTS", "HDG", "HAH", "KBC", "SSI", "MWG", "GAS", "REE", "VNM", "BID", "GMD", "DGC", "STB", "VJC"]

def _fetch_one(args):
    """Fetch 1 mã, tự retry 3 lần nếu gặp rate limit."""
    ticker, start_date, end_date = args
    for attempt in range(3):
        try:
            df = Quote(symbol=ticker, source='VCI').history(
                start=start_date, end=end_date, interval='1D'
            )
            if df is None or df.empty:
                return ticker, None

            df = df.reset_index()

            # --- Tìm cột GIÁ ĐÓNG CỬA (close) ---
            close_candidates = [c for c in df.columns
                                if c.lower() in ('close', 'closeprice', 'close_price', 'gia_dong_cua', 'c')]
            if not close_candidates:
                # Fallback: lấy cột numeric đầu tiên không phải ngày
                num_cols = [c for c in df.columns
                            if pd.api.types.is_numeric_dtype(df[c]) and c != date_col]
                print(f"[{ticker}] No close col found. All columns: {df.columns.tolist()}")
                if not num_cols:
                    return ticker, None
                close_col = num_cols[0]
                print(f"[{ticker}] Using fallback column: '{close_col}'")
            else:
                close_col = close_candidates[0]

            # --- Tìm cột NGÀY ---
            date_candidates = [c for c in df.columns
                               if c.lower() in ('time', 'date', 'trading_date', 'tradingdate',
                                                'datetime', 't', 'index', 'ngay')]
            if not date_candidates:
                print(f"[{ticker}] No date column found. Columns: {df.columns.tolist()}")
                return ticker, None
            date_col = date_candidates[0]

            # --- Parse ngày: thử string trước, fallback ms ---
            raw = df[date_col].copy()
            parsed = pd.to_datetime(raw, errors='coerce')           # thử string 'YYYY-MM-DD'
            if parsed.isna().mean() > 0.5:                          # nếu >50% NaT → thử ms
                parsed = pd.to_datetime(raw, unit='ms', errors='coerce')

            df[date_col] = parsed
            df = df.dropna(subset=[date_col])
            df = df.set_index(date_col)
            df.index = df.index.normalize()

            series = pd.to_numeric(df[close_col], errors='coerce').dropna()
            if series.empty:
                return ticker, None
            return ticker, series

        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in ['rate', '429', 'limit', 'quota']):
                wait = (attempt + 1) * 15
                time.sleep(wait)
            else:
                # In lỗi thật ra console để dễ debug
                print(f"[{ticker}] attempt {attempt+1} error: {e}")
                return ticker, None
    return ticker, None

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_data(tickers: tuple, years: int = 3):
    """
    Fetch theo batch nhỏ để tránh rate limit gói Guest (20 req/phút).
    - Batch 5 mã, nghỉ 15s giữa batch → tối đa ~16 req/phút.
    - max_workers=3 trong mỗi batch tránh burst cùng lúc.
    """
    end_date   = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=years * 365)).strftime('%Y-%m-%d')

    BATCH_SIZE  = 5
    BATCH_DELAY = 15
    price_df = {}
    failed_tickers = []

    batches = [tickers[i:i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]

    for batch_idx, batch in enumerate(batches):
        args = [(t, start_date, end_date) for t in batch]
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(_fetch_one, a): a[0] for a in args}
            for future in as_completed(futures):
                ticker, series = future.result()
                if series is not None:
                    price_df[ticker] = series
                else:
                    failed_tickers.append(ticker)
        # Nghỉ giữa batch (bỏ qua batch cuối)
        if batch_idx < len(batches) - 1:
            time.sleep(BATCH_DELAY)

    if not price_df:
        return pd.DataFrame(), failed_tickers
    result = pd.DataFrame(price_df).sort_index().ffill().bfill()
    result = result[result.index >= '2000-01-01']
    result.index = pd.to_datetime(result.index)
    return result, failed_tickers

def get_portfolio_stats(weights, mean_returns, cov_matrix):
    port_ret = np.sum(mean_returns * weights)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return port_ret, port_vol

# --- 3. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2010/2010684.png", width=60)
    st.header("⚙️ Cấu hình Chiến thuật")
    user_ticker = st.text_input("🔍 Mã cổ phiếu mục tiêu:", value="HPG").upper()

    strategy = st.radio("🎯 Mục tiêu tối ưu:",
                        ["Max Sharpe (Lợi nhuận/Rủi ro)", "Min Volatility (An toàn tối đa)"])

    use_top3 = st.toggle("🔝 Chỉ lấy Top 3 mã mạnh nhất", value=False,
                         help="Sau khi tối ưu, AI sẽ lọc ra 3 mã tốt nhất và chia lại tỷ trọng để danh mục tập trung hơn.")

    analyze_btn = st.button("🚀 Phân tích & Backtest", type="primary", use_container_width=True)
    st.divider()
    st.caption("Logic: Markowitz Optimization - Student ID: 11230980")

rf_rate = st.sidebar.slider("Lãi suất phi rủi ro (Risk-free rate)", 0.0, 0.1, 0.05, 0.01)

# --- 4. GIAO DIỆN CHÍNH ---
st.title("🤖 Robo-Advisor: Intelligent Portfolio Manager")
st.info("Hệ thống kết hợp thuật toán Markowitz từ nghiên cứu của bạn và giao diện Dashboard chuyên nghiệp.")

if analyze_btn and user_ticker:
    with st.status("🛠️ Đang chạy Engine tối ưu hóa...", expanded=True) as status:

        # --- Validate format mã (HOSE/HNX: 2-4 ký tự chữ hoa) ---
        import re as _re
        if not _re.match(r'^[A-Z]{2,5}[0-9]?$', user_ticker):
            status.update(label="Mã không hợp lệ!", state="error")
            st.error(f"❌ **'{user_ticker}'** không đúng định dạng mã chứng khoán VN (2–5 chữ cái, ví dụ: FPT, VCB, HPG).")
            st.stop()

        n_batches = -(-len(BASKET) // 5)
        est_seconds = n_batches * 15
        st.write(f"📡 Đang tải dữ liệu {len(BASKET)+1} mã theo batch (ước tính ~{est_seconds}s để tránh rate limit)...")

        scan_list = tuple([user_ticker] + [t for t in BASKET if t != user_ticker])
        df_all, failed = fetch_data(scan_list)

        # Debug: hiển thị columns thực tế nếu có lỗi
        if failed:
            with st.expander(f"⚠️ {len(failed)} mã không tải được — xem chi tiết"):
                st.write("Các mã lỗi:", failed)
                st.caption("Kiểm tra log terminal để xem tên cột thực tế từ vnstock.")

        if df_all.empty or user_ticker not in df_all.columns:
            status.update(label="Lỗi!", state="error")
            st.error(f"Không tìm thấy dữ liệu cho mã {user_ticker}. Kiểm tra lại mã hoặc chờ rate limit reset.")
        else:
            # 1. Tính toán lợi nhuận log
            returns_all = np.log(df_all / df_all.shift(1)).dropna()
            corr_matrix = returns_all.corr()
            best_hedges = corr_matrix[user_ticker].sort_values()[1:5].index.tolist()

            if len(best_hedges) < 2:
                status.update(label="Lỗi!", state="error")
                st.error("Không đủ dữ liệu để tạo danh mục đa dạng hóa. Thử lại sau hoặc đổi mã khác.")
                st.stop()

            final_list = [user_ticker] + best_hedges

            df_final = df_all[final_list]
            returns = np.log(df_final / df_final.shift(1)).dropna()

            # 2. Tham số Markowitz
            mu = returns.mean() * 252
            sigma = returns.cov() * 252
            num_assets = len(final_list)

            st.write("🧮 Đang giải bài toán tối ưu...")

            def objective(w):
                p_ret, p_vol = get_portfolio_stats(w, mu, sigma)
                if "Max Sharpe" in strategy:
                    return -(p_ret - rf_rate) / p_vol
                return p_vol

            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            res = minimize(objective, num_assets * [1. / num_assets],
                           method='SLSQP',
                           bounds=tuple((0, 0.3) for _ in range(num_assets)),
                           constraints=constraints)

            st.markdown("### 🥧 Chiến lược giải ngân (Tỷ trọng % Vốn)")
            st.caption("💡 *Tỷ trọng được giới hạn tối đa 30%/mã để đảm bảo đa dạng hóa.*")

            final_weights = res.x

            # 3. Logic lọc Top 3
            if use_top3:
                st.write("🔝 Đang lọc Top 3 mã tối ưu nhất...")
                w_series = pd.Series(final_weights, index=final_list)
                top3 = w_series.nlargest(3)
                final_weights = np.zeros(num_assets)
                for i, t in enumerate(final_list):
                    if t in top3.index:
                        final_weights[i] = top3[t] / top3.sum()

            status.update(label="✅ Đã tối ưu xong!", state="complete", expanded=False)

            # --- HIỂN THỊ KẾT QUẢ ---
            st.subheader(f"📊 Kết quả chiến lược: {strategy}")
            p_ret, p_vol = get_portfolio_stats(final_weights, mu, sigma)

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Lợi nhuận dự phóng", f"{p_ret * 100:.1f}%")
            k2.metric("Độ biến động (Rủi ro)", f"{p_vol * 100:.1f}%")
            k3.metric("Điểm Sharpe", f"{(p_ret - rf_rate) / p_vol:.2f}")

            st.markdown("⚠️ *Lưu ý: Backtest chạy trên dữ liệu lịch sử. Trọng số được tính toán dựa trên dữ liệu in-sample.*")
            port_returns = (returns * final_weights).sum(axis=1)
            port_cum = (1 + port_returns).cumprod()
            benchmark_cum = (1 + returns.mean(axis=1)).cumprod()
            k4.metric("Hiệu quả so với Benchmark", f"{(port_cum.iloc[-1] - benchmark_cum.iloc[-1]) * 100:+.1f}%")

            # --- BIỂU ĐỒ ---
            st.divider()
            tab1, tab2 = st.tabs(["📈 Hiệu quả Backtest", "🧬 Phân tích Kỹ thuật"])

            with tab1:
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=port_cum.index, y=port_cum, name="Danh mục AI (Của bạn)", line=dict(color='#00FFAA', width=3)))
                fig_bt.add_trace(go.Scatter(x=port_cum.index, y=benchmark_cum, name="Benchmark (Chia đều vốn)", line=dict(color='#808495', dash='dot')))
                fig_bt.add_trace(go.Scatter(x=port_cum.index, y=(1 + returns[user_ticker]).cumprod(), name=f"Chỉ giữ {user_ticker}", line=dict(color='rgba(255, 68, 68, 0.5)', dash='dot')))
                fig_bt.update_layout(title="So sánh tăng trưởng tài sản (Gốc = 1)", hovermode="x unified", template="plotly_dark")
                st.plotly_chart(fig_bt, use_container_width=True)

            with tab2:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### Ma trận tương quan")
                    st.plotly_chart(px.imshow(corr_matrix.loc[final_list, final_list], text_auto=".2f", color_continuous_scale='RdBu_r'), use_container_width=True)
                with c2:
                    st.markdown("#### Tỷ trọng giải ngân")
                    res_df = pd.DataFrame({'Mã': final_list, 'Tỷ trọng (%)': final_weights * 100}).sort_values('Tỷ trọng (%)', ascending=False)
                    st.dataframe(res_df[res_df['Tỷ trọng (%)'] > 0],
                                 column_config={"Tỷ trọng (%)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)},
                                 hide_index=True, use_container_width=True)

            st.chat_message("assistant").write(f"AI nhận định: Danh mục của bạn đã được tối ưu theo hướng {strategy}. Nhấn vào các mã để tra cứu thêm.")
