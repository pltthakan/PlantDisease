# **Bitki Hastalıkları Sınıflandırma Projesi – MobileNetV2 Eğitim Raporu**

## **1) Proje Amacı**

Bu proje, **bitki yaprağı görüntülerinden hastalık tespiti** yapmak için geliştirilmiştir. Kullanılan veri seti *PlantVillage*’dir. Model, derin öğrenme tabanlı **MobileNetV2** mimarisi ile transfer öğrenme yaklaşımını kullanmaktadır.

---

## **2) Kullanılan Teknolojiler ve Kütüphaneler**

* Python, TensorFlow / Keras
* MobileNetV2 (ImageNet ön eğitimli ağırlıklar)
* ImageDataGenerator (Veri artırma)
* Matplotlib / Seaborn (Görselleştirme)
* Scikit-Learn (Classification Report & Confusion Matrix)
* Grad-CAM (Model açıklanabilirliği)
* Google Colab GPU

---

## **3) Veri Seti ve Veri Ön İşleme**

Veri Kaggle’dan çekilmiş ve şu şekilde kullanılmıştır:

* Görüntüler **224x224** boyutuna ölçeklendi.
* Model girdisine uygun olarak `preprocess_input` fonksiyonu kullanıldı.
* **Veri artırma (augmentation):**

  * Döndürme (rotation_range=30)
  * Kaydırma (width/height_shift=0.2)
  * Zoom (zoom_range=0.2)
  * Yatay çevirme (horizontal_flip=True)
* Veri %80 eğitim, %20 doğrulama olarak otomatik ayrıldı.

Bu artırma işlemleri **overfitting’i azaltarak** modelin genelleme yeteneğini güçlendirir.

---

## **4) Model Mimarisi**

Model, **MobileNetV2 tabanlıdır** ve üstüne özel katmanlar eklenmiştir:

```
MobileNetV2 (include_top=False, weights='imagenet')
 ↓
| Katman                        | Amacı                                                         |
| ----------------------------- | ------------------------------------------------------------- |
| `GlobalAveragePooling2D`      | Çıkan 3D feature map’i özetler, parametre sayısını azaltır.   |
| `BatchNormalization`          | Eğitim stabilitesini artırır.                                 |
| `Dropout(0.5)`                | Overfitting’i azaltır.                                        |
| `Dense(256, ReLU)`            | Yeni özellik kombinasyonları öğrenir (tam bağlantılı katman). |
| `BatchNormalization`          | Aktivasyon dağılımını dengeler.                               |
| `Dropout(0.3)`                | Tekrar overfitting önler.                                     |
| `Dense(NUM_CLASSES, Softmax)` | Son çıktıları sınıf olasılıklarına dönüştürür.                |

```

* İlk aşamada **taban model dondurulmuştur (trainable=False)**.
* İkinci aşamada **son ~30 katman açılarak fine-tuning yapılmıştır**.

---

## **5) Eğitim Stratejisi (2 Aşama)**

### **Aşama 1 – Kafa Eğitimi**

* Sadece üst katmanlar eğitildi.
* Öğrenme oranı: `1e-3`
* Callback’ler:

  * ModelCheckpoint
  * EarlyStopping
  * ReduceLROnPlateau

### **Aşama 2 – Fine-Tuning**

* Taban modelin belirli kısmı eğitime açıldı.
* Öğrenme oranı düşürüldü: `1e-5`
* Sınıf dengesizliği varsa `class_weight` uygulandı.

Bu yaklaşım:
✔ Ağı daha stabil öğrenmeye zorlar
✔ Aşırı ezberlemeyi azaltır
✔ Doğruluğu artırır

---

## **6) Model Performansının Değerlendirilmesi**

Eğitim sonunda:

* doğrulama doğruluğu (`val_accuracy`)
* sınıflandırma raporu (precision, recall, F1-score)
* karışıklık matrisi (confusion matrix)

hesaplanmıştır.

Bu sayede:

* Hangi hastalık sınıflarının karıştığı,
* Modelin hangi bölgelerde güçlü/zayıf olduğu

analiz edilebilir.

---

## **7) Tek Görsel Testi**

Model `test_leaf.jpeg` görselini alır:

1. Ön işleme yapılır.
2. Model tahmin eder.
3. Tahmin edilen sınıf adı ve olasılık yüzdesi gösterilir.

Bu işlem modelin gerçek kullanım senaryosunu simüle eder.

---

## **8) Grad-CAM (Modeli Açıklama)**

Grad-CAM ile modelin **yaprak üzerinde hangi bölgelere odaklandığı** görselleştirilmiştir.

Bu:

* Modelin **gerçekten hastalık bölgelerini öğrendiğini**
* Rastgele tekstür/pattern ezberlemediğini

doğrulamak için önemlidir.

---

## **9) Olası Hatalar ve Çözüm Önerileri**

| Problem                   | Sebep                           | Çözüm                                    |
| ------------------------- | ------------------------------- | ---------------------------------------- |
| Overfitting               | Veri az / model güçlü           | Daha fazla augmentation + dropout arttır |
| Validation accuracy düşük | Fine-tuning katman sayısı fazla | `fine_tune_at` değerini küçült           |
| Sınıf karışması           | Görseller çok benzer            | Daha fazla veri veya MixUp/CutMix        |

---


