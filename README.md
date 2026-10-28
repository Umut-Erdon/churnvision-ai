# 📉 ChurnVision AI

Telekom müşterilerinde **kayıp (churn) riskini** tahmin eden, modern arayüzlü Streamlit dashboard'u.

- **Veri:** IBM Telco Customer Churn örnek veri seti (7.043 müşteri) — `data/telco_churn.csv`. Kaynak: IBM, herkese açık örnek veri; paylaşımlarda IBM'e atıf verin.
- **Model:** HistGradientBoosting · hold-out ROC-AUC ≈ 0.84
- **Risk puanları:** 5-fold out-of-fold tahmin (model, müşteriyi kendi puanını hesaplarken görmez)
- **Sayfalar:** Genel Bakış · Müşteri Riski (filtre + CSV indir) · Müşteri Detayı (gauge + aksiyon önerisi) · Model & Veri (metrikler, önem sırası, confusion matrix)

## Çalıştırma
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Ücretsiz yayınlama
GitHub'a yükle → [share.streamlit.io](https://share.streamlit.io) → repo'yu seç → `app.py` → Deploy. LinkedIn'e canlı linki koy.

## Sınırlar
Veri tek zaman kesitidir; zaman bazlı doğrulama yok, kampanya etkisi ölçülmedi.
