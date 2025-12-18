# 🎧 ML Algoritmalı Müzik Öneri Sistemi (Spotify API)

Bu proje, **Spotify API** kullanarak şarkı araması yapan ve farklı **makine öğrenmesi algoritmaları** ile benzer şarkılar öneren bir web uygulamasıdır.

Amaç:
- Aynı şarkı girdisi için
- Farklı algoritmalarla
- Farklı öneri sonuçları üretmek
- Ve bu sonuçları **benzerlik oranlarıyla karşılaştırmaktır**

---

## 🚀 Özellikler

- 🔍 Spotify üzerinden canlı şarkı arama
- 🎶 Spotify linki ile şarkıyı açabilme
- 🎧 Preview (varsa 30 sn dinleme)
- 📊 Algoritmaya göre değişen benzerlik oranı tablosu
- 🧠 Birden fazla ML algoritmasıyla öneri

---

## 🧠 Kullanılan Algoritmalar

Sistem, **metin tabanlı özellikler** (şarkı adı + sanatçı + albüm) üzerinden çalışır.

### 1️⃣ TF-IDF + Cosine Similarity
- Metinlerin ayırt edici kelimelerini çıkarır
- En temel ve referans algoritmadır

### 2️⃣ K-En Yakın Komşu (KNN)
- TF-IDF vektör uzayında
- Cosine mesafesine göre en yakın şarkıları bulur

### 3️⃣ SVD (TruncatedSVD / LSA)
- TF-IDF matrisini daha düşük boyuta indirger
- Gürültüyü azaltarak anlamsal benzerlik yakalar

### 4️⃣ Lineer Regresyon
- TF-IDF + SVD ile elde edilen vektörlerden
- Şarkılar arası **benzerlik skorunu tahmin eder**

### 5️⃣ Rastgele Orman (Random Forest)
- Doğrusal olmayan ilişkileri yakalamak için
- Çoklu karar ağaçlarıyla benzerlik skoru üretir

### 6️⃣ Sinir Ağı (MLP)
- Çok katmanlı yapay sinir ağı
- Şarkı çiftleri arasındaki benzerliği öğrenir

> Not: Spotify Audio Features endpoint’i (403) nedeniyle,
> tüm algoritmalar **metin tabanlı özellikler** ile çalışmaktadır.

---

## 🖥️ Kullanılan Teknolojiler

- Python 3
- Flask
- Spotify Web API
- scikit-learn
- NumPy
- HTML / CSS

---

## 📦 Kurulum

### 1️⃣ Projeyi klonla
```bash
git clone https://github.com/kullanici-adi/proje-adi.git
cd proje-adi
```

### 2️⃣ Sanal ortam oluştur
```bash
python -m venv .venv
```

### 3️⃣ Sanal ortamı aktif et

**Windows**
```bash
.venv\Scripts\activate
```

**Mac / Linux**
```bash
source .venv/bin/activate
```

### 4️⃣ Gerekli kütüphaneleri kur
```bash
pip install -r requirements.txt
```

---

## 🔑 Spotify API Ayarları

Proje klasöründe `.env` dosyası oluştur:

```env
SPOTIFY_CLIENT_ID=BURAYA_CLIENT_ID
SPOTIFY_CLIENT_SECRET=BURAYA_CLIENT_SECRET
```

Spotify Developer Dashboard:
https://developer.spotify.com/dashboard

---

## ▶️ Çalıştırma

```bash
python app.py
```

Tarayıcıdan aç:
```
http://127.0.0.1:5000
```

---

## 🧪 Sistem Nasıl Çalışır?

1. Kullanıcı bir şarkı arar
2. Spotify API’den sonuçlar alınır
3. İlk bulunan şarkı **referans şarkı** kabul edilir
4. Seçilen algoritmaya göre benzerlik skorları hesaplanır
5. En yüksek skorlu şarkılar önerilir
6. Sağ panelde yüzdelik benzerlik oranları gösterilir
7. Algoritma değiştikçe sonuçlar değişir

---

## 👥 Proje Ekibi

- Furkan Korunur
- Anıl Şahin
- Kerem Can Hayvacı

---

## ⚠️ Bilinen Kısıtlar

- Spotify `preview_url` her şarkı için mevcut değildir
- Spotify Audio Features endpoint’i bazı uygulamalarda erişime kapalıdır
- Bu nedenle sistem metin tabanlı çalışacak şekilde tasarlanmıştır

---

## 📌 Sonuç

Bu proje, farklı makine öğrenmesi algoritmalarının aynı problem üzerindeki etkisini
karşılaştırmalı ve görsel şekilde sunmak amacıyla geliştirilmiştir.

Eğitim ve akademik kullanım içindir.
