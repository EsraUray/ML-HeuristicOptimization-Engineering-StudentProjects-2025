
Makine Öğrenmesi Tabanlı Göl Seviyesi Tahmini

Bu proje, iklim değişikliği, artan su kullanımı ve kentleşme gibi faktörlerin tatlı su kaynakları üzerindeki etkilerini analiz etmek amacıyla göl su seviyelerinin makine öğrenmesi ve derin öğrenme yöntemleri ile tahmin edilmesini hedeflemektedir.

Beyşehir Gölü ve ABD’deki Beş Büyük Göl’e ait uzun dönemli gerçek ölçüm verileri kullanılmıştır.

Çözülen Mühendislik Problemi

Göl su seviyesi; ekosistem dengesi, tarım, sanayi ve içme suyu temini açısından kritik bir parametredir. Su seviyesindeki ani düşüşler kuraklık riskini artırırken, aşırı yükselmeler taşkınlara yol açabilmektedir.

Bu çalışmada amaç, farklı yapay sinir ağı ve derin öğrenme modellerini kullanarak göl su seviyelerini doğru ve güvenilir biçimde tahmin etmek ve hangi yöntemin daha başarılı olduğunu ortaya koymaktır.

Kullanılan Yöntemler ve Metodoloji

1. Çok Katmanlı Yapay Sinir Ağları (ÇKYSA)
Aşağıdaki eğitim algoritmaları test edilmiştir:
- Levenberg–Marquardt (LM)
- Conjugate Gradient (CGB, CGF, CGP)
- Scaled Conjugate Gradient (SCG)
- Resilient Backpropagation (RBP)
- Quasi-Newton (QN)
- One Step Secant (OSS)
- Variable Learning Rate Backpropagation (GDX)

2. Derin Öğrenme Mimarileri
- LSTM
- BiLSTM
- GRU
- BiGRU

 Veri seti %80 eğitim – %20 test olarak ayrılmıştır.  
 Modellerin performansı farklı gecikme senaryoları (T-1 … T-12) için değerlendirilmiştir.


Değerlendirme Kriterleri

Model performansı aşağıdaki metrikler kullanılarak ölçülmüştür:
RMSE (Kök Ortalama Kare Hata
MAE (Ortalama Mutlak Hata)
R² (Belirleme Katsayısı)

Elde Edilen Temel Sonuçlar ve Değerlendirme

- Levenberg–Marquardt tabanlı yapay sinir ağı, hızlı yakınsama ve yüksek doğruluk sağlaması nedeniyle referans model olarak öne çıkmıştır.
- Derin öğrenme modelleri özellikle uzun dönemli bağımlılıkları yakalamada başarılı sonuçlar üretmiştir.
- Tüm modeller aynı veri seti ile eğitilerek adil bir karşılaştırma yapılmıştır.
- Sonuçlar grafikler (zaman serisi, regresyon grafikleri) ile desteklenmiştir. 

Çalışma Alanı

- Türkiye: Beyşehir Gölü  
- ABD: Superior, Michigan, Huron, Erie ve Ontario Gölleri  

Bu iki farklı coğrafi yapı, model genellenebilirliğinin test edilmesini sağlamıştır.

Model / Demo

Bu çalışmada modeller MATLAB ortamında geliştirilmiştir.  
Herhangi bir web tabanlı deploy veya canlı demo bulunmamaktadır.

Öğrenciler
Emine Sena YAMAN 
  221453014@ogrenci.karatay.edu.tr

Hamdi KOÇ  
  221453025@ogrenci.karatay.edu.tr

KTO Karatay Üniversitesi  
İnşaat Mühendisliği Bölümü
