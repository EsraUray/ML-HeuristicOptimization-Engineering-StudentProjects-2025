# Milk-Run Tabanlı Üretim Hattında Kapasiteli Araç Rotalama Optimizasyonu

[cite_start]**Hazırlayanlar:** Birkan Sert, Nazım Kerem Tetik, Yiğit Yücetürk [cite: 6, 476]

## 1. Çözülen Mühendislik Probleminin Açıklaması
Bu proje, endüstriyel üretim hatlarında iç lojistik süreçlerinin verimliliğini artırmak amacıyla "Milk-Run" (döngüsel sefer) yönteminin optimize edilmesini konu alır. [cite_start]Problem, literatürde **Kapasite Kısıtlı Araç Rotalama Problemi (CVRP)** olarak tanımlanmıştır[cite: 29].

Temel amaç; [cite_start]10 farklı istasyona hizmet veren 2 araçlık bir filonun, kapasite kısıtlarına (Q=20) uyarak, tüm istasyonları en kısa mesafede ve en düşük karbon emisyonuyla ziyaret etmesini sağlamaktır[cite: 31, 37]. [cite_start]Bu problem NP-Hard (çözümü zor) sınıfında yer aldığı için klasik yöntemler yerine meta-sezgisel yaklaşımlar gerektirir[cite: 89].

## 2. Kullanılan Yöntem / Metodoloji
[cite_start]Problemin çözümü için **MATLAB** ortamında **Genetik Algoritma (GA)** geliştirilmiştir[cite: 494].
Kullanılan yaklaşımın temel adımları şunlardır:
* **Kodlama:** Her birey (kromozom) bir rotayı temsil eder.
* [cite_start]**Seçim (Selection):** Turnuva seçimi yöntemi ile iyi bireyler seçilir[cite: 81].
* [cite_start]**Çaprazlama (Crossover) ve Mutasyon:** Yeni rotalar üretmek ve yerel optimumdan kaçmak için Swap (yer değiştirme) mutasyonu uygulanır[cite: 85].
* [cite_start]**Ceza Fonksiyonu:** Kapasite veya araç sayısı kısıtını aşan çözümlere yüksek ceza puanı atanarak elenmeleri sağlanır[cite: 127].

## 3. Elde Edilen Temel Sonuçlar ve Değerlendirme
Geliştirilen algoritma, 200 iterasyonluk simülasyonlar sonucunda optimum rotayı başarıyla bulmuştur. İstatistiksel sonuçlar şöyledir:

| Metrik | Değer |
| :--- | :--- |
| **En İyi Toplam Mesafe** | [cite_start]**323.81 metre** [cite: 441] |
| **Toplam Karbon Emisyonu** | [cite_start]**259.05 gram CO2** [cite: 442] |
| **Kullanılan Araç Sayısı** | [cite_start]2 [cite: 442] |
| **Ortalama Maliyet** | [cite_start]0.2708 kg CO2 [cite: 443] |

[cite_start]Algoritma, rastgele dağıtılan başlangıç rotalarına kıyasla lojistik maliyetlerini minimize etmiş ve araç kapasitelerini (Araç 1: %85, Araç 2: %75 doluluk) verimli kullanmıştır[cite: 456].

## 4. Klasör Yapısı
* `src/`: Projenin kaynak kodları (MATLAB .m dosyaları).
* `notebooks/`: Analiz ve denemeler (Varsa Jupyter not defterleri).
* `model/`: Algoritma çıktıları veya kaydedilmiş model parametreleri.

---
[cite_start]*Bu proje KTO Karatay Üniversitesi Endüstri Mühendisliği Bölümü "Optimizasyon Teorisi" dersi kapsamında hazırlanmıştır.* [cite: 474, 477]
