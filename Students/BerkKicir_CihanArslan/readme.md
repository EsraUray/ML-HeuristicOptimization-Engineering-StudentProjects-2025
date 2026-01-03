# Harmoni Arama Algoritması Tabanlı Reaktif Güç Kompanzasyon Optimizasyonu

**Ders:** Makine Öğrenmesi ve Optimizasyon Mühendislik Uygulamaları
**Kurum:** KTO Karatay Üniversitesi - Mühendislik ve Doğa Bilimleri Fakültesi
**Dönem:** 2025 Bahar

## 👨‍🔬 Proje Grubu
* **Berk KICIR** (241451040)
* **Cihan ARSLAN** (231451030)
* **Danışman:** Dr. Öğr. Üyesi Esra URAY

---

## 📑 Proje Özeti (Abstract)
Elektrik güç sistemlerinde, endüktif ve kapasitif yüklerin dinamik değişimi, şebeke kararlılığını ve enerji verimliliğini doğrudan etkilemektedir. TEDAŞ regülasyonlarına göre, işletmelerin reaktif güç tüketimlerini belirli sınırlar (Endüktif <%20, Kapasitif <%15) içerisinde tutmaları zorunludur.

Geleneksel kompanzasyon röleleri, genellikle "sıralı anahtarlama" mantığıyla çalışmakta olup, kondansatör kademelerinin **ayrık (discrete)** ve düzensiz (non-uniform) olduğu karmaşık panolarda optimum anahtarlamayı sağlamada yetersiz kalabilmektedir.Bu çalışmada, türev gerektirmeyen stokastik bir optimizasyon yöntemi olan **Harmoni Arama Algoritması (Harmony Search Algorithm - HSA)** kullanılarak, **Kısıtlı Kombinatoryal Optimizasyon (Constrained Combinatorial Optimization)** problemi modellenmiş ve çözülmüştür.

Projenin temel amacı, cezalı amaç fonksiyonu (penalized cost function) yaklaşımıyla reaktif güç hatasını minimize etmek ve anahtarlama elemanlarının (kondansatörlerin) ömrünü uzatacak optimum kondansatör kombinasyonunu belirlemektir.

---

## ⚙️ Yöntem ve Algoritmik Tasarım

Problem, sürekli değişkenler yerine (0,1) durumlarını içeren ayrık bir çözüm uzayında tanımlanmıştır. Matlab ortamında geliştirilen algoritma, aşağıda belirtilen parametre seti ile işletilmiştir:

| Parametre | Sembol | Değer | Teknik Açıklama |
|---|---|---|---|
| **Harmoni Hafıza Boyutu** | $HMS$ | 60 | Çözüm uzayını tarayan popülasyon vektör büyüklüğü. |
| **Hafıza Kabul Oranı** | $HMCR$ | 0.98 | Algoritmanın mevcut iyi çözümleri koruma eğilimi (Exploitation). |
| **Ton Ayarlama Oranı** | $PAR$ | 0.45 | Yerel minimum tuzaklarından kaçış için kullanılan mutasyon olasılığı (Exploration). |
| **Maksimum İterasyon** | $MaxIter$ | 10,000 | Yakınsama kriteri. |

### Amaç Fonksiyonunun Matematiksel Modeli
Sistemin maliyet fonksiyonu ($Cost$), net reaktif güç hatası ve kısıt ihlalleri üzerinden aşağıdaki gibi formülize edilmiştir:

$$Min(f) = |Q_{net}| + P_{regülasyon} + P_{donanım}$$

Burada;
* **$Q_{net}$:** Hedeflenen (Yük) ve Gerçekleşen (Kondansatör) reaktif güç arasındaki fark.
* **$P_{regülasyon}$:** %20 Endüktif veya %15 Kapasitif sınırları aşıldığında uygulanan yüksek katsayılı ($10^5$) ceza fonksiyonu.
* **$P_{donanım}$:** Büyük güçlü kondansatörlerin gereksiz anahtarlamasını önlemek için uygulanan ağırlıklı ceza katsayısı ($steps^{1.5} \times 0.05$).

---

## 📊 Deneysel Bulgular ve Senaryo Analizi

Geliştirilen algoritmanın performansı iki farklı uç senaryo üzerinde test edilmiştir.

### Durum 1: Nominal Yüklenme ve Tam Kompanzasyon (Ideal Case)
Sistemin aktif güç talebinin 100 kW ve reaktif yükün 80 kVAr olduğu, donanım kapasitesinin yeterli olduğu durumdur.

* **Sistem Durumu:** Kararlı (Stable)
* **Sonuç:** Algoritma, çözüm uzayındaki global optimum noktayı tespit etmiştir. Büyük kademeler yerine "hassas" (küçük değerli) kondansatör gruplarına öncelik vererek **0.00 kVAr** hata ile sistemi dengeye oturtmuştur.
* **Teknik Çıkarım:** HSA, ayrık değişkenli sistemlerde türevsel yöntemlere ihtiyaç duymadan sıfır hataya yakınsayabilmektedir.

![Nominal Yük Analizi]
*Grafik 2: Nominal yük altında optimizasyon sürecinin yakınsama grafiği ve kapasitif bölge yerleşimi.*

### Durum 2: Doyum Bölgesi ve Yetersiz Kapasite (Saturation Case)
Bu senaryoda, işletmenin endüktif yük talebinin (300 kVAr), panodaki toplam kurulu gücü (207.5 kVAr) aştığı bir "arıza/yetersizlik" durumu simüle edilmiştir.

* **Sistem Durumu:** Doyum (Saturation)
* **Sonuç:** Fiziksel olarak tam kompanzasyonun imkansız olduğu bu durumda, algoritma **çökme (divergence)** yaşamamıştır. Mevcut tüm kondansatörleri devreye alarak hatayı fiziksel olarak mümkün olan en alt limit olan **%37** seviyesine çekmiştir.
* **Teknik Çıkarım:** Algoritma, kısıtların fiziksel olarak sağlanamadığı durumlarda dahi kararlı yapısını koruyarak "Best-Effort" (En iyi çaba) prensibiyle çalışmaktadır.

![Doyum Bölgesi Analizi]
*Grafik 1: Donanım sınırlarının zorlandığı senaryoda algoritmanın kararlılık analizi.*

---

## 📂 Dosya Yapısı
* `readme.md`: Proje hakkında bilgilendirme.
* `src/`: MATLAB kaynak kodu.
* `images/`: Simülasyon çıktılarına ait grafiksel veriler.

---
*Bu çalışma, TÜBİTAK 2209-A Üniversite Öğrencileri Araştırma Projeleri Destekleme Programı standartlarına uygun olarak, akademik araştırma metodolojisi çerçevesinde gerçekleştirilmiştir.*
