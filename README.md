# BLM5103 - Bilgisayarla Görme Ödevi 1

Bu proje, bir video akışında hazmat (tehlikeli madde) işaretlerini ve kırmızı/mavi varilleri tespit eden Python tabanlı bir görüntü işleme sistemidir.

##  Proje Yapısı ve Açıklamaları

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
