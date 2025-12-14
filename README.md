# 📊 Sales Forecasting Dashboard (Streamlit)

Dashboard analisis dan forecasting penjualan berbasis **time series harian**
menggunakan pendekatan statistik klasik dan machine learning ringan.

Aplikasi ini dirancang agar:
- Stabil di **Streamlit Community Cloud (gratis)**
- Modular untuk pengembangan lanjutan (local / production)

---

## 🚀 Fitur Utama
- Load data langsung dari **Google Sheets (CSV export)**
- Filter:
  - Rentang tanggal
  - Multi-produk
- Analisis:
  - Preview data
  - Descriptive statistics
  - Correlation analysis
  - Time series forecasting
- Preprocessing:
  - Outlier handling (IQR clipping)
  - Log transform (opsional)
  - Rolling smoothing
- Forecasting models:
  - Naive
  - Moving Average
  - SES (Simple Exponential Smoothing)
  - Holt
  - Holt-Winters
  - ARIMA
  - Linear Regression
  - Random Forest
- Auto model selection berdasarkan **RMSE**
- Forecast masa depan (7–180 hari)

> ⚠️ Model berat seperti LSTM, XGBoost, dan Prophet **dinonaktifkan untuk cloud gratis**
> dan hanya disarankan untuk penggunaan lokal / server sendiri.

---

## 🧱 Struktur Project
