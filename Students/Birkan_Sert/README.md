# Milk-Run Tabanlı Üretim Hattında Kapasiteli Araç Rotalama Optimizasyonu

**Hazırlayanlar:** Birkan Sert, Nazım Kerem Tetik, Yiğit Yücetürk

## 1. Çözülen Mühendislik Probleminin Açıklaması

Bu proje, endüstriyel üretim hatlarında iç lojistik süreçlerinin verimliliğini artırmak amacıyla "Milk-Run" (döngüsel sefer) yönteminin optimize edilmesini konu alır. Problem, literatürde **Kapasite Kısıtlı Araç Rotalama Problemi (CVRP)** olarak tanımlanmıştır.

Temel amaç; 10 farklı istasyona hizmet veren 2 araçlık bir filonun, kapasite kısıtlarına (Q=20) uyarak, tüm istasyonları en kısa mesafede ve en düşük karbon emisyonuyla ziyaret etmesini sağlamaktır. Bu problem NP-Hard (çözümü zor) sınıfında yer aldığı için klasik yöntemler yerine meta-sezgisel yaklaşımlar gerektirir.

## 2. Kullanılan Yöntem / Metodoloji

Problemin çözümü için **MATLAB** ortamında **Genetik Algoritma (GA)** geliştirilmiştir.
Kullanılan yaklaşımın temel adımları şunlardır:

- **Kodlama:** Her birey (kromozom) bir rotayı temsil eder.
- **Seçim (Selection):** Turnuva seçimi yöntemi ile iyi bireyler seçilir.
- **Çaprazlama (Crossover) ve Mutasyon:** Yeni rotalar üretmek ve yerel optimumdan kaçmak için Swap (yer değiştirme) mutasyonu uygulanır.
- **Ceza Fonksiyonu:** Kapasite veya araç sayısı kısıtını aşan çözümlere yüksek ceza puanı atanarak elenmeleri sağlanır.

## 3. Elde Edilen Temel Sonuçlar ve Değerlendirme

Geliştirilen algoritma, 200 iterasyonluk simülasyonlar sonucunda optimum rotayı başarıyla bulmuştur. İstatistiksel sonuçlar şöyledir:

| Metrik                     | Değer               |
| :------------------------- | :------------------ |
| **En İyi Toplam Mesafe**   | **323.81 metre**    |
| **Toplam Karbon Emisyonu** | **259.05 gram CO2** |
| **Kullanılan Araç Sayısı** | 2                   |
| **Ortalama Maliyet**       | 0.2708 kg CO2       |

Algoritma, rastgele dağıtılan başlangıç rotalarına kıyasla lojistik maliyetlerini minimize etmiş ve araç kapasitelerini (Araç 1: %85, Araç 2: %75 doluluk) verimli kullanmıştır.

## 4. Klasör Yapısı

- `src/`: Projenin kaynak kodları.
- `notebooks/`: Analiz ve denemeler.
- `model/`: Algoritma çıktıları veya kaydedilmiş model parametreleri.
- `dataset/`: Projede kullanılan istasyonlar ve talep miktarları.

---

_Bu proje KTO Karatay Üniversitesi Endüstri Mühendisliği Bölümü "Optimizasyon Teorisi" dersi kapsamında hazırlanmıştır._
