 # Tarımsal Alanlarda Hektar Başına Ton Cinsinden Verimin Tahmin Edilmesi

## 1. Çözülen Problem
Proje tarımsal alanlarda ton cinsinden verimini; çevresel faktörlere, iklim koşullarına ve tarımsal uygulanan yöntemler doğrultusunda tahmin etmeyi amaçlamaktadır.

## 2. Yöntem / Metodoloji

1) Öncelikle veri setine veri temizleme işlemleri uygulanmıştır. Bu aşamada uygulanan yöntemler şunlardır:
- Veri setindeki sütunların veri tiplerinin kontrol edilmesi,
- Veri setinde eksik (missing) değerlerin tespit edilmesi,
- Tutarsız metinler ve yazım hatalarının kontrol edilmesi,
- Yinelenen (duplicate) kayıtların kontrol edilmesi
- Aykırı değerlerin (outlier) incelenmesi

2) Veri temizleme aşamasından sonra keşifçi veri analizi ile değişkenler arasındaki ilişkiler analiz edilmiştir. Bu aşamada uygulanan yöntemler şunlardır:
- Sayısal değişkenlerin dağılımlarının incelenmesi için histogram çizilmesi,
- Sayısal değişkenler arasındaki korelasyonun incelenmesi için corr() fonksiyonunun kullanılması,

3) Temizlenmiş veri seti daha sonra Streamlit ile kullanıcıdan veri alıp modelleme amacıyla kayıt edildi.

4) Modelleme aşamasından önce, modelin verileri daha iyi anlayabilmesi için farklı veri tiplerine sahip değişkenlere ColumnTransformer yapısı kullanıldı. Bu kapsamda:
- Sayısal değişkenler, ölçeklemek için StandardScaler() kullanıldı.
- Boolean değişkenler, herhangi bir işlem uygulanmadı.
- Kategorik değişkenler, OneHotEncoder yöntemiyle kategorik sütunlar sayısala dönüştürüldü.
- Kullanıcıdan alınan girdiler, eğitim sürecinde kullanılmış olan ColumnTransformer ile dönüştürüldü.

5) Model eğitimi ve tahmin yaptırma kapsamında:
- Veri ön işleme adımlarının ardından, bağımsız değişkenlere sabit terim (constant) eklenip **Ordinary Least Squares (OLS)** yöntemi ile doğrusal regresyon modeli eğitildi.
- Kullanıcıdan alınıp ColumnTransformer ile dönüştürülen girdiler modelde tahmin için kullanıldı.

6) Model çıktılarının kullanıcıya sunulması
- Streamit kullanılarak kullanıcıdan alınan girdiler ile model tahmin edildi ve bu hektar başına ton cinsinden tahmin edilen değer kullanıcıya gösterildi.
- Tahmin edilen değerin yanı sıra kullanıcı girdileri, kullanıcı girdilerinin encode edilmiş hali, modelin özeti ve modelin metrikleri kullanıcıya sunulmuştur.




## 3. Sonuçlar

Varsayılan parametrelerle tahmin edilen modelin metrikleri:

|Metrik | Değer |
|-------|-------|
|MAE    | 0.39816096770164294|
|RMSE   | 0.5024668497161627|
|R2     | 0.9126182603171531|

Model yüksek açıklayıcılığa sahip, güvenilir bir performans sergilemektedir.


## 4. Klasör Yapısı
```
Omer_Faruk_Keskin/
├── README.md
├── src/
│   └── Yield.py
└── notebooks/
    └── Main.ipynb
```

## 5. Veri Seti
Link: https://www.kaggle.com/datasets/samuelotiattakorah/agriculture-crop-yield


