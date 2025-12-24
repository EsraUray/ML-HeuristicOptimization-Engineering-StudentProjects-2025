# Konya İli Sıcaklık Tahmini ve Makine Öğrenmesi Uygulaması
# (Temperature Prediction for Konya Province using Machine Learning)

---

## 🇹🇷 Proje Raporu (TR)

### 1. Proje Özeti ve Amacı
Bu çalışmanın temel amacı, Konya iline ait 2002-2023 yılları arasındaki meteorolojik verileri kullanarak, farklı Makine Öğrenmesi (ML) algoritmalarının performanslarını karşılaştırmak ve en başarılı model ile 2024-2025 yılları için aylık ortalama sıcaklık tahminleri yapmaktır. Sıcaklık değişimlerinin modellenmesi; tarım, enerji yönetimi ve şehir planlaması gibi alanlarda stratejik öneme sahiptir.

### 2. Metodoloji ve Veri İşleme
Projede Meteoroloji Genel Müdürlüğü'nden temin edilen 21 yıllık veri seti kullanılmıştır. Veriler modele verilmeden önce şu aşamalardan geçirilmiştir:
* **Veri Ön İşleme:** Eksik verilerin kontrolü ve `MinMaxScaler` kullanılarak verilerin 0-1 aralığına normalize edilmesi (Yapay Sinir Ağları performansı için kritik).
* **Kayan Pencere (Sliding Window):** Zaman serisi analizi için son 12 ayın verisi girdi (input), bir sonraki ayın verisi çıktı (output) olacak şekilde veri seti dönüştürülmüştür.
* **Eğitim/Test Ayrımı:** Verinin %90'ı eğitim, son 24 ayı (2022-2023) test seti olarak ayrılmıştır.

### 3. Kullanılan Modeller ve Hiperparametreler
Performans karşılaştırması için 4 farklı regresyon modeli eğitilmiştir:

1.  **Linear Regression:** Temel eğilim (trend) analizi için referans model (OLS Yöntemi).
2.  **SVR (Destek Vektör Regresyonu):**
    * *Kernel:* RBF (Radial Basis Function)
    * *C (Ceza Katsayısı):* 100
    * *Gamma:* 0.1
3.  **Random Forest Regressor:**
    * *Ağaç Sayısı (n_estimators):* 100
    * *Random State:* 42
4.  **MLP (Multi-Layer Perceptron - Yapay Sinir Ağı):**
    * *Gizli Katmanlar:* (100, 50) nörondan oluşan 2 katman.
    * *İterasyon:* 2000
    * *Aktivasyon:* ReLU

### 4. Sonuçlar ve Değerlendirme
Modellerin başarısı R² (Belirleme Katsayısı), RMSE (Kök Ortalama Kare Hata) ve MAE (Ortalama Mutlak Hata) metrikleri ile ölçülmüştür. Test verisi üzerindeki sonuçlar aşağıdadır:

| Model | R² Score | MAE (°C) | RMSE |
|-------|----------|----------|------|
| MLP (Yapay Sinir Ağı) | **0.937** | **1.82** | **2.18** |
| Random Forest | 0.935 | 1.78 | 2.22 |
| SVR (RBF Kernel) | 0.931 | 1.81 | 2.29 |
| Linear Regression | 0.930 | 1.88 | 2.30 |

Yapılan analizler sonucunda **MLP (Yapay Sinir Ağı)** modeli, en yüksek R² ve en düşük RMSE değerine sahip olduğu için "En Başarılı Model" olarak seçilmiştir. Bu model tüm veri setiyle tekrar eğitilerek 2024 ve 2025 yılları için sıcaklık tahminleri üretilmiştir.

---

## 🇬🇧 Project Report (EN)

### 1. Project Description
The main objective of this study is to compare the performance of different Machine Learning (ML) algorithms using meteorological data of Konya province between 2002-2023 and to predict monthly average temperatures for the years 2024-2025 using the best performing model. Accurate temperature forecasting is crucial for sectors such as agriculture and energy management.

### 2. Methodology and Data Preprocessing
A 21-year dataset obtained from the General Directorate of Meteorology was utilized. The following preprocessing steps were applied:
* **Normalization:** Data was scaled to the 0-1 range using `MinMaxScaler` to improve Neural Network convergence.
* **Sliding Window Algorithm:** The dataset was transformed for time-series forecasting, where the past 12 months are used to predict the next month.
* **Train/Test Split:** The first 90% of the data was used for training, while the last 24 months (2022-2023) were reserved for testing.

### 3. Models and Hyperparameters
Four different regression models were trained for comparison:

1.  **Linear Regression:** Used as a baseline model for trend analysis.
2.  **SVR (Support Vector Regression):** Configured with RBF kernel, C=100, and Gamma=0.1.
3.  **Random Forest Regressor:** An ensemble method with 100 estimators.
4.  **MLP (Multi-Layer Perceptron):** An Artificial Neural Network with 2 hidden layers (100, 50 neurons) and ReLU activation function.

### 4. Results and Discussion
Model performance was evaluated using R², RMSE, and MAE metrics. The results on the test set are as follows:

| Model | R² Score | MAE (°C) | RMSE |
|-------|----------|----------|------|
| **MLP (ANN)** | **0.937** | **1.82** | **2.18** |
| Random Forest | 0.935 | 1.78 | 2.22 |
| SVR (RBF Kernel) | 0.931 | 1.81 | 2.29 |
| Linear Regression | 0.930 | 1.88 | 2.30 |

The **MLP (Artificial Neural Network)** model was selected as the best model due to having the highest R² score and the lowest RMSE. Consequently, it was used to generate future temperature predictions for 2024 and 2025.
