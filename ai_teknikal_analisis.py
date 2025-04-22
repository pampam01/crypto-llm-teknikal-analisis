import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import ollama
import tempfile
import base64
import os
import streamlit as st

from openai import OpenAI


client = OpenAI(api_key="sk-464e48554f864482845263d410373153", base_url="https://api.deepseek.com")


# set up aplikasi streamlit
st.set_page_config(layout="wide")
st.title("AI Teknikal Analisis untuk Crypto")
st.sidebar.header("Pengaturan")


# input data crypto dan range nya 
ticker = st.sidebar.text_input("Ticker", "BTC-USD")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2025-01-01"))


# ambil data crypto
if st.sidebar.button("Fetch Data"):
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    if not data.empty:
        # Periksa dan tampilkan data awal
        st.write("Data yang diambil:")
        st.write(data.head())  # Menampilkan beberapa baris pertama
        st.write("Kolom yang tersedia:", data.columns)  # Menampilkan kolom yang tersedia

        # Tangani NaN dengan mengisi atau menghapus
        # Akses kolom dengan MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        available_columns = data.columns.intersection(["Open", "High", "Low", "Close"])
        data = data.dropna(subset=available_columns)  # Menghapus baris yang mengandung NaN di kolom yang tersedia

        st.session_state["crypto_data"] = data
        st.success("Data berhasil diambil dan diproses!")
    else:
        st.error("Data tidak tersedia untuk rentang tanggal yang dipilih.")

# cek apakah data sudah diambil
if "crypto_data" in st.session_state:
    data = st.session_state["crypto_data"]

    # tampilkan data crypto
    if not data.empty:
        fig = go.Figure(data=[
            go.Candlestick(
                x=data.index,
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                name="Candlestick"
            )
        ])

        # sidebar: pilih teknikal indikator 
        st.sidebar.subheader("Pilih Teknikal Indikator")
        indicators = st.sidebar.multiselect(
            "Teknikal Indikator",
            ["20-Day SMA", "20-Day EMA", "20-Day Bollinger Bands", "VWAP"],
            default=["20-Day SMA"]
        )

        # tambahkan teknikal indikator
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
                bb_upper = sma + 2 * std
                bb_lower = sma - 2 * std
                fig.add_trace(go.Scatter(x=data.index, y=bb_upper, mode='lines', name='BB Upper'))
                fig.add_trace(go.Scatter(x=data.index, y=bb_lower, mode='lines', name='BB Lower'))
            elif indicator == "VWAP":
                data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
                fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], mode='lines', name='VWAP'))

        for indicator in indicators:
            add_indicator(indicator)
        
        fig.update_layout(
            xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(fig)
    else:
        st.error("Data tidak tersedia setelah diproses.")

    # analisis chart dengan deepsek
   # analisis chart dengan ollama
st.subheader("Analisis Chart dengan Ollama AI")
if st.button("Jalankan AI Teknikal Analisis"):
    with st.spinner("Menjalankan AI..."):
        # Buat folder baru untuk menyimpan file sementara
        temp_dir = os.path.join(tempfile.gettempdir(), "crypto_analysis_temp")
        os.makedirs(temp_dir, exist_ok=True)

        # Simpan chart ke temp file di folder baru
        with tempfile.NamedTemporaryFile(suffix=".png", dir=temp_dir, delete=False) as temp_file:
            fig.write_image(temp_file.name, scale=0.5)  # Mengurangi resolusi gambar
            tmpfile_path = temp_file.name

        # Baca gambar dan encode ke base64
        with open(tmpfile_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")

        # Persiapan AI analysis request dengan ollama
        messages = [{
            'role': 'user',
            'content': """You are a Stock Trader specializing in Technical Analysis at a top financial institution.
                        Analyze the stock chart's technical indicators and provide a buy/hold/sell recommendation.
                        Base your recommendation only on the candlestick chart and the displayed technical indicators.
                        First, provide the recommendation, then, provide your detailed reasoning.
                """,
            'images': [image_data]
        }]

        try:
            # Menggunakan Ollama untuk analisis
            response = ollama.chat(model="gemma3:4b", messages=messages)

            # Periksa struktur respons
            if "messages" in response and len(response["messages"]) > 0:
                last_message = response["messages"][-1]
                if "content" in last_message:
                    ai_response = last_message["content"]
                    # Tampilkan respons dengan format yang lebih baik
                    st.markdown("**Response AI Teknikal Analisis**")
                    st.markdown(ai_response)
                else:
                    st.error("Respons dari Ollama tidak mengandung 'content'.")
            else:
                st.error("Respons dari Ollama tidak mengandung 'messages' atau kosong.")
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

        # Hapus temp file
        os.remove(tmpfile_path)