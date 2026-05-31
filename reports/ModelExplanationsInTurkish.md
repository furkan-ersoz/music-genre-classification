# Model Mimarileri: EfficientNet-B0 + Temporal Attention ve ResNet18 + BiGRU

---

## Model001 — EfficientNet-B0 + Temporal Attention

### Temel Fikir

Bir müzik parçasının türünü belirlemek için insanlar ne yapar? Sadece "genel tona" bakmazlar — belirli anlara odaklanırlar. Bir gitar rifu, bir davul patlaması, bir bas çizgisi. `model001`, tam da bu sezgiyi matematikleştiren bir mimari kurar.

Model iki büyük bileşenden oluşuyor: **EfficientNet-B0** adlı bir görsel özellik çıkarıcı ve onun üstüne yerleştirilen bir **temporal attention** (zamansal dikkat) katmanı.

---

### EfficientNet-B0: Neden Bu Backbone?

Spectrogramlar, ses sinyallerinin fotoğrafıdır — yatay eksende zaman, dikey eksende frekans, piksel yoğunluğunda enerji vardır. Yani aslında bir görüntü sınıflandırma problemiyle karşı karşıyayız; bu da ImageNet'te eğitilmiş görsel modelleri başlangıç noktası olarak kullanmayı mantıklı kılar.

EfficientNet-B0, verimlilik/performans dengesindeki öne çıkan modellerden biri. Kod `freeze_until=0` ile gelir (varsayılan); yani tüm katmanlar ince ayara açık. Alternatif olarak `freeze_until=4` gibi bir değerle erken katmanlar dondurulup sadece derin katmanlar öğrenmeye bırakılabilir — bu, veri azsa overfitting'i engeller.

Backbone çıkışı `(B, 1280, H', W')` boyutlu bir özellik haritasıdır. Burada 1280, öğrenilmiş kanal sayısı; H' ve W' ise uzamsal boyutlardır (giriş 224×224 ise yaklaşık 7×7).

---

### Zamansal Boyuta Geçiş

Bir spectrogramın genişliği zamana karşılık gelir. Özellik haritasının genişlik boyutunu da "zaman adımları" olarak yorumlarsak, CNN'in çıkardığı özellikleri bir dizi (sequence) olarak işleyebiliriz.

`model001` tam bunu yapar:

```
(B, 1280, H', W')
    → AdaptiveAvgPool2d((7, 1))   # genişliği 1'e indir
    → (B, 1280, 7, 1)             # 7 "zaman dilimi" kaldı
    → squeeze + permute
    → (B, 7, 1280)                # 7 adımlı bir dizi
    → Linear projection
    → (B, 7, 256)                 # daha küçük boyuta indir
```

Artında elimizde spektrogramı 7 parçaya bölen bir dizi var. Her parça, o zaman dilimininin CNN tarafından çıkarılmış özellik vektörü.

---

### Multi-Head Self-Attention

İki katman `TemporalAttention` uygulanıyor. Her katman şunu yapıyor: her zaman adımı, diğer tüm zaman adımlarıyla "konuşuyor" ve hangisinin kendisiyle ne kadar ilgili olduğunu öğreniyor. Sonuçta bazı zaman dilimlerine daha fazla, bazılarına daha az ağırlık veriliyor.

Bu mekanizma çok kafadan (`num_heads=4`) oluşuyor; her kafa farklı bir "ilişki örüntüsü" öğrenebilir. Bir kafa ritimdeki tekrarları, diğeri melodik yapıyı, bir diğeri dinamik değişimleri yakalayabilir.

Son adımda 7 zaman adımı ortalamalanıyor (`mean(dim=1)`) ve bir sınıflandırıcı kafasına veriliyor.

---

### Neden Tercih Edilir?

- **Transfer learning avantajı:** Sıfırdan başlamak yerine milyonlarca görüntüde öğrenilmiş zengin özelliklerden faydalanır.
- **Uzun menzilli bağımlılıklar:** Saf CNN'ler lokal örüntüleri yakalar; attention ise spektrogramın başı ile sonu arasındaki ilişkileri modelleyebilir.
- **Yorumlanabilirlik:** Attention ağırlıklarına bakarak "model bu parçanın neresine odaklandı?" sorusu sorulabilir.
- **Verimli boyut seçimi:** 1280'den 256'ya projeksiyon, attention hesaplama maliyetini önemli ölçüde düşürür.

---

## Model002 — ResNet18 + Bidirectional GRU + Attention Pooling

### Temel Fikir

`model002`, müzikal zaman serisinin *yönlü* ve *seçici* yorumlanmasına odaklanıyor. İki bileşen bu amaca hizmet ediyor: **BiGRU** (iki yönlü tekrarlayan ağ) ve **attention pooling** (dikkat tabanlı toplama).

---

### ResNet18: Neden Farklı Bir Backbone?

`model001` EfficientNet'i kullanırken `model002` ResNet18'i tercih ediyor. ResNet18 daha basit ve daha eski bir mimari ama birkaç avantajı var:

- **Daha az parametre** — hızlı eğitim ve daha az overfitting riski
- **Özellik haritası boyutu daha kontrollü** — son katman 512 kanal üretiyor, bu GRU'ya vermek için ideal
- **Bilinen davranış** — spectrogramlarda iyi çalıştığı geniş çaplı olarak doğrulanmış

Kodda `backbone = nn.Sequential(*list(backbone.children())[:-2])` ile son iki katman (avgpool ve fc) çıkarılıyor. Sadece özellik çıkarıcı kısmı kullanılıyor: çıkış `(B, 512, H', W')`.

---

### Frekans Boyutunu Çöktürmek

Spectrogramda dikey eksen frekans, yatay eksen zamandır. `model002` şunu yapıyor: frekans boyutunu `AdaptiveAvgPool2d((1, None))` ile tek bir vektöre indirgiyor. Bu sayede:

```
(B, 512, H', W')
    → (B, 512, 1, W')    # frekans ortalaması
    → squeeze → permute
    → (B, W', 512)       # W' adet zaman adımı, her biri 512-boyutlu
```

Artık elimizde, spektrogramın her zaman dilimine karşılık gelen 512 boyutlu özellik vektörlerinden oluşan bir dizi var.

---

### Bidirectional GRU: İki Yönlü Anlama

GRU (Gated Recurrent Unit), LSTM'in daha hafif versiyonu. Bir "gizli durum" taşıyarak dizinin üzerinden geçer ve her adımda önceki adımların etkisini seçici biçimde saklayıp atar.

**Bidirectional** (iki yönlü) olması kritik: bir GRU dizinin başından sonuna, diğeri sonundan başına gidiyor. İkisinin çıkışları birleştiriliyor (`gru_hidden * 2`). Bu sayede model, "şu andaki zaman dilimini" hem geçmiş bağlamla hem de gelecek bağlamla değerlendirebiliyor. Örneğin bir caz parçasında saksofon solosunun *öncesinde* ve *sonrasında* ne olduğu, o an için ipucu verebilir.

Kod iki katman GRU (`gru_layers=2`) kullanıyor; birinci katmanın çıkışı ikinci katmana giriyor, bu da daha karmaşık zaman örüntüleri öğrenilmesini sağlıyor.

---

### Attention Pooling: Hangi Anlara Odaklan?

BiGRU'nun tüm zaman adımları için ürettiği vektörler `(B, W', hidden*2)` boyutunda. Bunları tek bir temsile indirgemek için iki seçenek var:

1. **Ortalama al** — hepsini eşit ağırlıkla birleştir (model001'in yaptığı)
2. **Attention pooling** — her zaman adımına önem skoru ver, ağırlıklı topla (model002'nin yaptığı)

`AttentionPooling` modülü bir `Linear(hidden, 1)` katmanı kullanıyor: her zaman adımını bir skalere dönüştürüp softmax ile normalleştiriyor. Bu skalerin yüksek olduğu anlar daha fazla ağırlık alıyor. Model, hangi "anların" tür kararı için belirleyici olduğunu *öğreniyor*.

---

### model001 ile Karşılaştırma

| Özellik | model001 | model002 |
|---|---|---|
| Backbone | EfficientNet-B0 (1280 ch) | ResNet18 (512 ch) |
| Zamansal işlem | Multi-head self-attention | Bidirectional GRU |
| Toplama | Basit ortalama | Attention pooling |
| Bağlam yönü | Her zaman adımı herkesle | İki yönlü sıralı akış |
| Parametre sayısı | Daha fazla | Daha az |
| Güçlü olduğu yer | Küresel örüntüler, uzun bağımlılıklar | Sıralı gelişim, ritim, faz geçişleri |

---

### Tasarım Felsefesi

Her iki model de aynı problemi farklı "bilişsel stratejilerle" çözüyor. Bunu şöyle özetlemek mümkün:

**model001** bir fotoğrafa bakar gibi çalışıyor: tüm spektrogramı aynı anda görür, hangi bölgelerin birbirleriyle ilgili olduğunu self-attention ile hesaplar. Bu küresel bir bakış açısıdır — "bu parçanın genel tonal kimliği ne?"

**model002** bir müzisyen gibi dinler: sesi zaman içinde takip eder, ne geldiğini ve ne geleceğini birlikte değerlendirir. BiGRU'nun iki yönlü yapısı sayesinde "bu anın bağlamı" hem geçmişten hem gelecekten beslenir. Ardından attention pooling, "hangi anlara odaklanmalıyım?" sorusunu yanıtlar.

Pratikte bu iki modeli birlikte çalıştırıp sonuçları karşılaştırmak (ya da ensemble yapmak) güçlü bir strateji olur — zaten `run_all.py` bunu `models_list` içine her ikisini de ekleyerek destekliyor.
