# 🛒 Prediksi Jumlah Pembelian dengan XGBoost

Aplikasi web untuk memprediksi jumlah pembelian pelanggan menggunakan model XGBoost.

## 📋 Prerequisites

- Python 3.8+
- Dataset: `shopping_behavior_updated.csv`
- Git

## 🚀 Cara Deploy ke Streamlit Cloud

### Langkah 1: Persiapan Local

1. **Clone atau buat repository baru di GitHub**
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/USERNAME/REPOSITORY_NAME.git
git push -u origin main
```

2. **Train dan simpan model**
```bash
python train_and_save_model.py
```

Ini akan menghasilkan 3 file:
- `xgb_model.pkl` - Model XGBoost
- `label_encoder.pkl` - Label encoder untuk shipping_type
- `feature_columns.pkl` - Daftar kolom fitur

### Langkah 2: Struktur Folder

Pastikan struktur folder Anda seperti ini:
```
your-project/
│
├── app.py                          # File utama Streamlit
├── train_and_save_model.py         # Script untuk training model
├── requirements.txt                # Dependencies
├── .gitignore                      # File yang diabaikan Git
├── README.md                       # Dokumentasi
│
├── shopping_behavior_updated.csv   # Dataset (opsional)
├── xgb_model.pkl                   # Model terlatih
├── label_encoder.pkl               # Encoder
└── feature_columns.pkl             # Kolom fitur
```

### Langkah 3: Push ke GitHub

```bash
git add .
git commit -m "Add model files"
git push
```

⚠️ **PENTING**: Jika file `.pkl` terlalu besar (>100MB), gunakan Git LFS:
```bash
git lfs install
git lfs track "*.pkl"
git add .gitattributes
git commit -m "Add Git LFS"
git push
```

### Langkah 4: Deploy di Streamlit Cloud

1. Buka [share.streamlit.io](https://share.streamlit.io)
2. Login dengan akun GitHub Anda
3. Klik "New app"
4. Pilih:
   - **Repository**: Repository GitHub Anda
   - **Branch**: main
   - **Main file path**: app.py
5. Klik "Deploy!"

### Langkah 5: Tunggu Deployment

Streamlit Cloud akan:
- Clone repository Anda
- Install dependencies dari `requirements.txt`
- Menjalankan aplikasi

Proses ini biasanya memakan waktu 2-5 menit.

## 🔧 Testing Local

Sebelum deploy, test aplikasi di local:

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

Aplikasi akan terbuka di `http://localhost:8501`

## 📦 File yang Diperlukan

### 1. `app.py`
File utama aplikasi Streamlit dengan UI interaktif.

### 2. `requirements.txt`
Dependencies yang diperlukan:
- streamlit
- pandas
- numpy
- scikit-learn
- xgboost
- plotly

### 3. Model Files (`.pkl`)
- `xgb_model.pkl`: Model XGBoost terlatih
- `label_encoder.pkl`: Label encoder untuk encoding
- `feature_columns.pkl`: Daftar nama kolom

### 4. `train_and_save_model.py`
Script untuk melatih dan menyimpan model.

## 🎯 Fitur Aplikasi

- ✅ Input interaktif untuk semua fitur pelanggan
- ✅ Prediksi real-time jumlah pembelian
- ✅ Visualisasi gauge chart dan bar chart
- ✅ Analisis perbandingan dengan rata-rata
- ✅ Insight otomatis berdasarkan prediksi
- ✅ Responsive design

## 🐛 Troubleshooting

### Error: "Model file not found"
**Solusi**: Pastikan file `.pkl` sudah di-push ke GitHub dan ada di repository.

### Error: "Module not found"
**Solusi**: Pastikan semua dependencies ada di `requirements.txt` dengan versi yang benar.

### App loading sangat lambat
**Solusi**: 
- Pastikan file model tidak terlalu besar
- Gunakan Git LFS untuk file besar
- Optimize model size jika perlu

### Error saat prediksi
**Solusi**: 
- Pastikan semua input telah diisi
- Check bahwa feature columns sesuai dengan training
- Verify label encoder sudah loaded dengan benar

## 📊 Cara Menggunakan Aplikasi

1. Buka aplikasi di browser
2. Isi semua input di sidebar kiri:
   - Data demografis (usia, gender, lokasi)
   - Informasi produk (item, kategori, ukuran, warna)
   - Perilaku pembelian (frekuensi, riwayat)
   - Status promosi (diskon, kode promo, langganan)
3. Klik tombol "🔮 Prediksi Jumlah Pembelian"
4. Lihat hasil prediksi dan analisis

## 🔒 Keamanan

- Jangan push file `.env` atau credentials
- Gunakan `.gitignore` untuk file sensitif
- Jika menggunakan API keys, gunakan Streamlit Secrets

## 📝 Update Model

Untuk update model:
1. Train ulang model dengan data baru
2. Run `train_and_save_model.py`
3. Push file `.pkl` yang baru ke GitHub
4. Streamlit Cloud akan auto-redeploy

## 🤝 Kontribusi

Silakan buat pull request atau issue untuk improvements!

## 📄 Lisensi

MIT License

## 👨‍💻 Kontak

Untuk pertanyaan atau support, silakan buat issue di repository ini.

---

**Selamat Mencoba! 🚀**
