# CogniTicket Route – Türkçe destek yönlendirici

Bu paket, sizin eğittiğiniz dört modeli Streamlit uygulamasında çalıştırır. Ana yönlendirme distillation sonrası dinamik INT8 öğrenci BERT ile yapılır. Dört modelin ham kategorisi, skoru, sabit eşiği ve yönlendirmesi aynı mesaj için yan yana gösterilir. Bir mesaj hiçbir gerçek destek sistemine gönderilmez. Tahminler tamamen yerel CPU'da hesaplanır; çalışma sırasında model indirilmez.

## Windows'ta ilk kurulum

Python 3.12 (64 bit) önerilir; 64 bit Intel/AMD CPU gerekir. Python kurulurken Add Python to PATH seçili olmalı. Ağır modellerin açılması için birkaç GB boş RAM gerekir. GPU gerekmez.

1. Bu ZIP'yi Tümünü Ayıkla ile çıkarın.
2. İçindeki cogniticket_app klasörünü açın. Adres çubuğuna cmd yazıp Enter'a basın.
3. Şu komutları sırayla çalıştırın:

```bat
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-torch.txt
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

4. Daha önce indirdiğiniz şu beş ZIP'yi archives klasörüne kopyalayın; adlarını değiştirmeyin:

- baseline_stage2.zip
- word2vec_nn_stage2.zip
- bert_teacher_stage2.zip
- bert_student_int8_stage2.zip
- final_test_stage2.zip

5. Aynı terminalde modelleri bir kez hazırlayın:

```bat
.venv\Scripts\python.exe prepare_models.py
```

6. Uygulamayı başlatın:

```bat
.venv\Scripts\python.exe -m streamlit run app.py
```

Sonraki açılışlarda run.bat dosyasına çift tıklayabilirsiniz. Tarayıcı açılmazsa http://localhost:8501 adresini açın. İlk analizde modeller yüklenir; sonraki analizlerde önbellek kullanılır. Colab'a geçmek gerekmez.

## Linux / macOS (x86-64)

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-torch.txt
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python prepare_models.py
.venv/bin/python -m streamlit run app.py
```

Apple Silicon / ARM için bu dosyalardaki x86 quantization motoru doğrudan desteklenmeyebilir; bu paket x86-64 CPU içindir.

## Uygulama davranışı

- Regex normalizasyonu Unicode NFC ve boşluk düzenidir. Olumsuzluklar, sayılar ve büyük/küçük harfler korunur.
- CAM-204 veya DW-6534 gibi model kodları regex ile çıkarılır; kodlar ürün/model doğrulaması yapılmış kayıtlar değildir.
- Edit distance ürün sözlüğünde tek ve bir düzenleme uzaklığındaki adaylara dar kapsamlı öneri verir. Çekimli her Türkçe ürün adını düzeltmesi beklenmez. Kutucuk işaretlenmedikçe öneri uygulanmaz. Bu teknik istatistiksel model değildir.
- ELIZA tarzı kurallar yalnızca tam selamlaşma/küçük sohbet girdilerini yanıtlar. “Merhaba, cihazım bozuk” sınıflandırıcıya gider. Selamlaşma ekranında model tahmini uydurulmaz.
- Aynı seçilmiş metin dört modele gider. BERT girdisi eğitimdeki gibi en fazla 64 tokendır. Uzun girdide kesilme uyarısı görünür.
- Her modelin kendi validation eşiği dondurulmuştur. INT8, FP32 öğrencinin eşiğini devralır. Arayüzden eşik ayarlanmaz. Skor kalibre edilmiş başarı olasılığı değildir. İnsan temsilciye ayrılması, kapsam dışı tespitin kusursuz olduğu anlamına gelmez.

## Sonuçların anlamı

reports içindeki tablolar gönderdiğiniz arşivlerden aynen kopyalanmıştır. Uygulama arşivdeki aynı tabloları gösterir. Kaynak: sizin Colab çalıştırmalarınız. 120 eğitim ve toplam 20 test örneği (10 destek, 10 kapsam dışı); tamamı ChatGPT üretimi yapay örneklerdir. Öğrencinin elle topladığı veya gerçek müşteri kaydı olarak sunulamaz. İngilizce destek ölçülmedi. %100 destek sınıflandırma doğruluğu yalnızca 10 mesaja aittir; bütün sistemin %100 başarısı değildir.

Final tabloda boyut öğrenilmiş ağırlıklar ve öğrenilmiş ön işleme dosyalarının boyutudur; tokenizer/config/raporlar hariçtir. CPU gecikmesi 10 ısınma sonrası 100 tekrar ortalamasıdır; metin ön işleme ve tokenization dahil, model yükleme ve arayüz hariçtir. Tek canlı tahmin süresi bu ortalama değildir.

Canlı uygulamaya eklenen regex ve opsiyonel yazım düzeltmesi eski raporun ölçüm hattında yoktu. Eski tablo korunmuş deneydir, yeni uygulamanın tüm işlem hattının tekrar ölçüldüğü iddia edilmez. Test görülmüş olduğundan bu testle uygulama/eşik ayarı yaparak tekrar tekrar yeni başarı iddiası üretilmemelidir. Yeni uçtan uca değerlendirme gerekiyorsa bağımsız, önceden görülmemiş ve insan incelemeli yeni test gerekir. Mimari diyagramda bu fark açıkça belirtilmelidir.

## Model kayıtları ve uyumluluk

Orijinal Colab kayıtlarında Transformers 4.57.1, Gensim 4.4.0, scikit-learn 1.6.1 ve PyTorch 2.14.1 kullanılmıştır. CPU kurulumunda aynı PyTorch sürümü hedeflenir. Gensim'in kaydettiği __main__.stable_hash başvurusu yüklemeden önce sağlanır. PyTorch state_dict dosyaları weights_only=True ve strict yüklemeyle açılır. Model arşivleri yalnızca kendi ürettiğiniz güvenilir dosyalar olmalıdır; joblib ve Gensim pickle tabanlıdır.

quantize_dynamic için DeprecationWarning alınabilir: mevcut kayıtlar eski eager dynamic INT8 yoluyla üretilmiştir. Runtime'ı rastgele yükseltmeyin; başka quantization API'sine geçiş yeni model kaydı ve doğrulama gerektirir. Uygulama sessizce başka model veya motor kullanmaz.

Yeniden yükleme/regresyon kontrolü (eski ham girdilerde kayıtlı kararları karşılaştırır, eğitim yapmaz):

```bat
.venv\Scripts\python.exe verify_models.py
```

## Depo teslimi için kalanlar

Bu bir uygulama paketidir; teslimin tamamı değildir. Güncel Colab eğitim notebook'unuzu notebooks klasörüne eklemelisiniz. Sunum (8–12 slayt), sesli iki dakikalık demo, veri/AI kaynak açıklaması ve erişilebilir model indirme bağlantıları ayrıca hazırlanmalıdır. Büyük ağırlıklar ve ZIP arşivleri GitHub deposuna eklenmemeli; kendi Hugging Face/Drive bağlantılarınız README'ye eklenmeli. Bu pakette sizin adınıza dışarıya dosya yayımlanmadı veya bağlantı uydurulmadı. Arşivler erişilebilir bağlantıdan indirildikten sonra hazırlama ve tek çalıştırma komutu yukarıdadır.

## AI kullanımı

Gemini: ilk MASSIVE deneyi ve ilk arayüz taslağı. ChatGPT: gereklilik incelemesi, yapay destek verileri, revize eğitim/distillation/quantization kodu, ölçüm kodu ve bu uygulama. Eğitimi ve raporlanan ölçümleri öğrenci Colab'da çalıştırdı. Sunumda bunlar açıkça belirtilmeli.
