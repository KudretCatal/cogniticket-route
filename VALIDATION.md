# Yerel doğrulama – 15 Kasım 2026

- Python 3.12; PyTorch 2.14.1+cpu; Transformers 4.57.1; Gensim 4.4.0; scikit-learn 1.6.1; Streamlit 1.51.0.
- Gönderilen beş arşiv açıldı; dört canlı model yerel dosyalardan yüklendi. Ağdan model indirilmedi.
- Orijinal rapordaki ham test girdileriyle 80 model–mesaj karşılaştırması yapıldı (ek FP32 öğrenci bu uygulamanın dört model kontrolüne dahil değil). 80 son yönlendirme kararı eşleşti. Bu bir tekrar yükleme/regresyon kontrolüdür; bağımsız yeni test değildir, orijinal sonuçlar değiştirilmedi.
- Bir reddedilmiş INT8 mesajının ham kategorisi orijinal kayıttan farklıydı; iki durumda da insan temsilciye gitti. Orijinal değerlendirme batch=4, canlı uygulama batch=1 çalışıyor; dinamik INT8 sayısal sonuçları batch/donanım farkından etkilenebilir. Bu fark için eşik veya ağırlık değiştirilmedi.
- Streamlit AppTest: başlangıç ekranı, boş giriş uyarısı, “Merhaba” kural cevabı ve “Kameram Wi-Fi ağına bağlanmıyor, model CAM-204” girdisinde dört modelin çalışması başarılı. INT8 ana yönlendirmesi Bağlantı ve Kurulum oldu.
- Regex model kodu çıkarma, boşluk normalizasyonu, olumsuzluğun korunması, karışık selamlaşma+destek mesajının kurala takılmaması ve ürün yazım önerisi kontrol edildi. Python kaynakları derlendi.
- Tarayıcı ekran görüntüsüyle görsel doğrulama tamamlanmadı; headless tarayıcı indirilemedi. Streamlit'in test runtime'ında etkileşim kontrolleri yapıldı.
- Windows'ta ayrıca çalıştırma yapılmadı. Kurulum yönergeleri Windows için hazırlanmıştır; esas runtime kontrolü Linux x86-64 CPU'da yapıldı.
