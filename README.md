# BLM5103 - Bilgisayarla Görme Ödevi 1
Bu proje, bir video akışında hazmat (tehlikeli madde) işaretlerini ve kırmızı/mavi varilleri tespit eden Python tabanlı bir görüntü işleme sistemidir.
##  Proje Yapısı 

```
24011124_odev1/
├── detector/                             # Nesne tespiti yapan modüller
│   ├── barrel_detector.py               # Renk eşikleme ile kırmızı/mavi varil tespiti yapan modül
│   └── orb_hazmat_detector.py           # SIFT + FLANN algoritması ile hazmat işareti tespiti yapan modül
│
├── utils/                                # Yardımcı fonksiyonlar
│   ├── draw_utils.py                    # Tespit edilen nesneleri çerçeve içerisine çizmek için kullanılır
│   └── iou_nms.py                       # IoU hesaplama ve Non-Maximum Suppression (NMS) uygulamaları
│
├── data/                                 # Girdi verilerinin bulunduğu dizin
│   └── hazmats/                         # 15 adet hazmat işareti görselinin bulunduğu klasör
│
├── main.py                               # Ana yürütülebilir dosya: video okuyup tespitleri yapan ve kullanıcı arayüzünü yöneten dosya
├── requirements.txt                      # Kullanılan Python kütüphanelerinin listesi
└── README.md                             # Bu döküman
```

## Gereksinimler

Python 3.8+ ve aşağıdaki kütüphaneler gereklidir:

```bash
pip install -r requirements.txt
```

## Çalıştırma

Projenin ana dosyası `main.py`’dir. Aşağıdaki komut ile çalıştırılır:

```bash
python main.py --video ./data/tusas-odev1.mp4 --log ./logs/output.log
```

> Not: `--log` parametresi opsiyoneldir.

### Hazmat İşaretleri (15 Adet)
- Explosives
- Blasting Agents
- Flammable Gas
- Non Flammable Gas
- Oxygen
- Fuel Oil
- Dangerous When Wet
- Flammable Solid
- Spontaneously Combustible
- Oxidizer
- Organic Peroxide
- Inhalation Hazard
- Poison
- Radioactive
- Corrosive

### Variller
- Kırmızı Varil
- Mavi Varil

## Duraklatma Özelliği
- Yeni bir nesne türü tespit edildiğinde video otomatik olarak duraklar.
- Devam etmek için herhangi bir tuşa (q hariç) basılır.
- Aynı obje tekrar görünse de bir daha durmaz.

# BLM5103 - Bilgisayarla Görme Ödevi 1

### 1. Giriş
Bu çalışma kapsamında, sahne üzerinde bulunan tehlikeli madde işaretleri (hazmat) ile kırmızı ve mavi renkli varillerin tespitine yönelik bir bilgisayarla görme sistemi geliştirilmiştir. Sistem, görüntü işleme ve özellik eşleme tekniklerini bir araya getirerek gerçek zamanlı analiz gerçekleştirmektedir.

### 2. Amaç
Geliştirilen sistemin temel amacı, video verileri üzerinde sahneye yerleştirilmiş olan 15 farklı hazmat işaretini ve renk kodlu varilleri doğru şekilde tespit ederek, duruma göre kullanıcıyı duraklatmak ve loglama işlemlerini gerçekleştirmektir.

### 3. Kullanılan Yöntemler

#### 3.1 Hazmat Tespiti (orb_hazmat_detector.py)
Hazmat işaretlerinin tespiti, SIFT (Scale-Invariant Feature Transform) ile yapılmış, ardından FLANN (Fast Library for Approximate Nearest Neighbors) algoritması ile eşleştirme gerçekleştirilmiştir. Her bir template görseli için RANSAC tabanlı homografi çıkarımı yapılmış, başarılı eşleşmeler sonucunda sahnedeki konumları belirlenmiştir. False positive oranını düşürmek amacıyla Non-Maximum Suppression (NMS) uygulanmıştır.

#### 3.2 Renk Bazlı Varil Tespiti (barrel_detector.py)
HSV renk uzayında belirlenmiş eşik değerleri kullanılarak kırmızı ve mavi variller maskeleme yöntemiyle tespit edilmiştir. Morfolojik işlemlerle gürültü azaltılmış, kontur analizi ile belirli büyüklükteki objeler filtrelenmiş, boy/en oranı ile varil benzerliği test edilmiştir.

#### 3.3 Çizim ve Görselleştirme (draw_utils.py)
Tespit edilen nesnelerin çerçeve içine alınarak kullanıcının görsel olarak bilgilendirilmesi sağlanmıştır. `cv2.rectangle` ve `cv2.putText` fonksiyonları ile her bir nesne ekranda gösterilmiştir.

#### 3.4 IoU ve NMS (iou_nms.py)
Tespitlerin üst üste binmesi durumunda en yüksek skora sahip kutunun korunması için IoU (Intersection over Union) değeri hesaplanmış ve `non_max_suppression()` fonksiyonu uygulanmıştır.

### 4. Ana Akış (main.py)
Bu dosya, tüm sistemin ana çatısını oluşturmaktadır. `argparse` ile video ve log dosyası alınmakta, her frame’de varil tespiti, belirli aralıklarla da hazmat tespiti yapılmaktadır. Yeni bir nesne türü algılandığında video duraklatılır ve terminale bilgi basılır.

### 5. Sonuç
Bu proje, bilgisayarla görme temelleri kullanılarak sahne analizi gerçekleştiren, gerçek zamanlı bir nesne tespiti sisteminin başarılı bir uygulamasıdır. Sistem, ödev yönergesindeki tüm gereklilikleri karşılamakta olup; modüler, genişletilebilir ve kullanıcı dostu bir yapıya sahiptir.
