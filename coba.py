import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import ollama
import tempfile
import base64
import os
import streamlit as st

# Konfigurasi aplikasi Streamlit
st.set_page_config(layout="wide")
st.title("AI Teknikal Analisis untuk Crypto")
st.sidebar.header("Pengaturan")

# Input pengguna
ticker = st.sidebar.text_input("Ticker", "BTC-USD")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2025-01-01"))

# Fungsi untuk mengambil data crypto dengan penanganan error
@st.cache_data
def fetch_crypto_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end, auto_adjust=False)

        # Jika data kosong, kembalikan pesan error
        if data.empty:
            st.error(f"Tidak ada data tersedia untuk {ticker}. Periksa ticker atau rentang tanggal.")
            return pd.DataFrame()

        # Definisikan kolom yang diperlukan
        required_columns = ["Open", "High", "Low", "Close"]

        # Cek kolom yang hilang
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            st.error(f"Error: Kolom yang hilang dalam data: {', '.join(missing_columns)}")
            return pd.DataFrame()

        # Pastikan indeksnya berupa datetime dan reset index jika perlu
        if not isinstance(data.index, pd.DatetimeIndex):
            data.reset_index(inplace=True)

        return data.dropna(subset=required_columns)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat mengambil data: {e}")
        return pd.DataFrame()

# Ambil dan simpan data
if st.sidebar.button("Fetch Data"):
    data = fetch_crypto_data(ticker, start_date, end_date)
    if not data.empty:
        st.session_state["crypto_data"] = data
        st.success("Data berhasil diambil!")
    else:
        st.error("Data tidak tersedia atau kolom tidak lengkap.")

# Proses data jika tersedia
if "crypto_data" in st.session_state:
    data = st.session_state["crypto_data"]

    # Buat candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        name="Candlestick"
    )])

    # Pilih teknikal indikator
    st.sidebar.subheader("Pilih Teknikal Indikator")
    indicators = st.sidebar.multiselect(
        "Teknikal Indikator",
        ["20-Day SMA", "20-Day EMA", "20-Day Bollinger Bands", "VWAP"],
        default=["20-Day SMA"]
    )

    # Fungsi untuk menambahkan indikator
    def add_indicator(indicator):
        if indicator == "20-Day SMA":
            sma = data['Close'].rolling(window=20).mean()
            fig.add_trace(go.Scatter(x=data.index, y=sma, mode='lines', name='SMA (20)'))
        elif indicator == "20-Day EMA":
            ema = data['Close'].ewm(span=20).mean()
            fig.add_trace(go.Scatter(x=data.index, y=ema, mode='lines', name='EMA (20)'))
        elif indicator == "20-Day Bollinger Bands":
            sma = data['Close'].rolling(window=20).mean()
            std = data['Close'].rolling(window=20).std()
            fig.add_trace(go.Scatter(x=data.index, y=sma + 2 * std, mode='lines', name='BB Upper'))
            fig.add_trace(go.Scatter(x=data.index, y=sma - 2 * std, mode='lines', name='BB Lower'))
        elif indicator == "VWAP":
            if "Volume" in data.columns and not data["Volume"].isnull().all():
                data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
                fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], mode='lines', name='VWAP'))
            else:
                st.warning("VWAP tidak tersedia karena data volume tidak lengkap.")

    # Tambahkan indikator ke chart
    for indicator in indicators:
        add_indicator(indicator)

    fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)

    # Analisis AI dengan Ollama
    st.subheader("Analisis Chart dengan Ollama AI")
    if st.button("Jalankan AI Teknikal Analisis"):
        with st.spinner("Menjalankan AI..."):
            try:
                # Simpan chart sebagai gambar sementara
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                    fig.write_image(temp_file.name, scale=0.5)
                    with open(temp_file.name, "rb") as image_file:
                        image_data = base64.b64encode(image_file.read()).decode("utf-8")

                # Permintaan analisis ke Ollama
                messages = [{
                    'role': 'user',
                    'content': """You are a Stock Trader specializing in Technical Analysis at a top financial institution.
                                Analyze the stock chart's technical indicators and provide a buy/hold/sell recommendation.
                                Base your recommendation only on the candlestick chart and the displayed technical indicators.
                                First, provide the recommendation, then, provide your detailed reasoning.""",
                    'images': [image_data]
                }]

                response = ollama.chat(model="gemma3:4b", messages=messages)
                ai_response = response.get("messages", [{}])[-1].get("content", "Tidak ada respons.")
                st.markdown("**Respons AI Teknikal Analisis**")
                st.markdown(ai_response)

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
            finally:
                os.remove(temp_file.name)

else:
    st.info("Silakan ambil data terlebih dahulu.")
