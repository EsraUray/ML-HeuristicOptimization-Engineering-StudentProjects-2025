# Türkiye Deprem Analizi ve Risk Tahmin Sistemi

Bu proje, Türkiye genelindeki deprem verilerini canlı olarak analiz eden ve hem makine öğrenmesi hem de istatistiksel yöntemler kullanarak deprem risk skorları üreten bir mühendislik çalışmasıdır.

## 📌 Çözülen Mühendislik Problemi
Türkiye'nin deprem kuşağında yer alması nedeniyle, geçmiş verilerden yola çıkarak geleceğe dönük risk analizi yapmak hayati önem taşımaktadır. Bu çalışma; belirli bir koordinat çevresinde kısa vadeli (30 gün) ve uzun vadeli (10 yıl) deprem olasılıklarını hesaplayarak, şehir bazlı "Bileşik Nihai Risk Skoru" üretmeyi hedefler.

## 🛠️ Kullanılan Yöntem ve Metodoloji
Projede üç temel bileşen birleştirilmiştir:

1.  **Kısa Vadeli Risk (Makine Öğrenmesi):**
    * **Algoritma:** CatBoost Classifier.
    * **Hedef:** Belirli bir bölgede 30 gün içinde M≥4 büyüklüğünde bir deprem olma olasılığı.
    * **Doğrulama:** TimeSeriesSplit (Zaman serisi bölme) yöntemi kullanılmıştır.
2.  **Uzun Vadeli Tehlike (İstatistiksel):**
    * **Yaklaşım:** Poisson Olasılık Dağılımı.
    * **Hedef:** 10 yıllık periyotta M≥6 büyüklüğünde deprem gerçekleşme ihtimali.
3.  **Jeolojik Bileşen:**
    * Şehir merkezlerinin diri fay hatlarına olan mesafeleri (Haversine mesafesi) üzerinden bir "Fay Segment Riski" hesaplanmıştır.

## 📊 Veri Kaynağı
* **Canlı Veri:** Kandilli Rasathanesi Canlı Deprem API'si kullanılarak veriler anlık güncellenmektedir.
* **Tarihsel Veri:** 1933'ten günümüze kadar olan Türkiye deprem kayıtları ön işleme tabi tutulmuştur.

## 🚀 Elde Edilen Sonuçlar
* Model, Türkiye genelindeki deprem aktivitesini harita üzerinde (Folium) görselleştirebilmektedir.
* CatBoost modeli ile yapılan testlerde deprem öncü işaretleri ve zamansal özelliklerin tahminleme gücü analiz edilmiştir.
* **Demo:** Kullanıcı bir şehir ismi girdiğinde sistem; o şehrin koordinatlarını, fay hattına mesafesini ve hesaplanan bileşik risk yüzdesini çıktı olarak vermektedir.

## 📂 Klasör Yapısı
- `notebooks/`: Projenin ana kodlarını içeren Jupyter Notebook dosyası.
- `dataset/`: Veri setinin bulunduğu dosya.

---
**Hazırlayan:** Utku Ceylan  
**Öğrenci Numarası:** 221450079
