import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import ollama
import streamlit as st

# st.set_page_config(layout="wide")


from openai import OpenAI





client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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"
)

# Set up aplikasi Streamlit
st.set_page_config(layout="wide")
st.title("AI Teknikal Analisis untuk Crypto")
st.sidebar.header("Pengaturan")

# Input data crypto dan range-nya 
ticker = st.sidebar.text_input("Ticker", "BTC-USD")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2025-01-01"))

# Tentukan jumlah data terakhir yang digunakan untuk analisis
num_data_points = st.sidebar.number_input("Jumlah Data Terakhir untuk AI", min_value=10, max_value=100, value=30, step=5)

# Ambil data crypto
if st.sidebar.button("Fetch Data"):
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    if not data.empty:
        st.write("Data yang diambil:")
        st.write(data.head())
        st.write("Kolom yang tersedia:", data.columns)

        # Tangani NaN
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        available_columns = data.columns.intersection(["Open", "High", "Low", "Close"])
        data = data.dropna(subset=available_columns)

        st.session_state["crypto_data"] = data
        st.success("Data berhasil diambil dan diproses!")
    else:
        st.error("Data tidak tersedia untuk rentang tanggal yang dipilih.")

# Cek apakah data sudah diambil
if "crypto_data" in st.session_state:
    data = st.session_state["crypto_data"]

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

        # Sidebar: pilih indikator teknikal 
        st.sidebar.subheader("Pilih Teknikal Indikator")
        indicators = st.sidebar.multiselect(
            "Teknikal Indikator",
            ["20-Day SMA", "20-Day EMA", "20-Day Bollinger Bands", "VWAP"],
            default=["20-Day SMA"]
        )

        # Tambahkan indikator teknikal
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

    # Analisis dengan AI
    st.subheader("Analisis Chart dengan Ollama AI")
    if st.button("Jalankan AI Teknikal Analisis"):
        with st.spinner("Menjalankan AI..."):

            # Ambil jumlah data yang ditentukan oleh pengguna
            latest_data = data.tail(num_data_points)[["Open", "High", "Low", "Close"]].to_string()

            # Ambil indikator yang dipilih
            indicator_results = ""

            if "20-Day SMA" in indicators:
                sma = data['Close'].rolling(window=20).mean().tail(num_data_points).to_string()
                indicator_results += f"\nSMA 20 Hari:\n{sma}\n"

            if "20-Day EMA" in indicators:
                ema = data['Close'].ewm(span=20).mean().tail(num_data_points).to_string()
                indicator_results += f"\nEMA 20 Hari:\n{ema}\n"

            if "20-Day Bollinger Bands" in indicators:
                sma = data['Close'].rolling(window=20).mean()
                std = data['Close'].rolling(window=20).std()
                bb_upper = (sma + 2 * std).tail(num_data_points).to_string()
                bb_lower = (sma - 2 * std).tail(num_data_points).to_string()
                indicator_results += f"\nBollinger Bands:\nUpper:\n{bb_upper}\nLower:\n{bb_lower}\n"

            if "VWAP" in indicators:
                data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
                vwap = data['VWAP'].tail(num_data_points).to_string()
                indicator_results += f"\nVWAP:\n{vwap}\n"

            # Prompt AI dengan candlestick + indikator teknikal
            messages = [{
                'role': 'user',
                'content': f"""Anda adalah seorang Trader Kripto yang mengkhususkan diri dalam Analisis Teknikal.
                              Berikut adalah {num_data_points} data candlestick terakhir dari {ticker}:
                              {latest_data}

                              Indikator teknikal yang digunakan:
                              {indicator_results}

                              Berdasarkan pola harga dan tren teknikal, apakah ini saat yang baik untuk membeli, menahan, atau menjual?
                              Berikan rekomendasi di awal, lalu jelaskan alasannya secara singkat.""",
            }]

            model_name = "thudm/glm-4-32b:free"  # Bisa ganti ke "gemma3:4b" jika ingin model lebih besar

            try:
                # Jalankan AI untuk analisis teknikal
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages
                )
                st.write(response.choices[0].message.content)

                # # Cek apakah respons memiliki struktur yang benar
                # if "message" in response and "content" in response["message"]:
                #     st.write("**Rekomendasi AI:**")
                #     st.write(response["message"]["content"])
                # else:
                #     st.error("Format respons dari Ollama tidak sesuai.")
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")


