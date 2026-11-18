from pathlib import Path
import json
import pandas as pd
import streamlit as st
from preprocessing import inspect_text

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='CogniTicket Route', page_icon='🌐',layout='wide')
st.markdown('''<style>
.stApp {background: radial-gradient(ellipse at 85% 0%,#132b42 0%,#0b1120 55%); color:#e8f0fa;}
.block-container {max-width:1180px; padding-top:3rem;}
h1 {letter-spacing:-1.5px;} .eyebrow{color:#38bdf8;letter-spacing:2px;font-size:12px;font-weight:700;}
.subtitle {color:#9fb1c9;font-size:17px;max-width:760px;line-height:1.7;}
[data-testid="stTextArea"] textarea{background:#111e30;color:#edf6ff;border:1px solid #324961;border-radius:14px;}
div.stButton>button,div.stFormSubmitButton>button{border-radius:10px;border-color:#38bdf8;}
[data-testid="stMetricValue"]{color:#73d4ff;}
[data-testid="stVerticalBlockBorderWrapper"]{border-radius:16px;}
</style>''',unsafe_allow_html=True)
st.markdown('<div class="eyebrow">AKILLI EV · MÜŞTERİ DESTEĞİ</div>',unsafe_allow_html=True)
st.title('🌐 CogniTicket Route')
st.markdown('<p class="subtitle">Destek talebinizi yazın. Dört modelin kararlarını karşılaştırın; ana yönlendirmeyi hafifletilmiş INT8 model yapsın.</p>',unsafe_allow_html=True)
st.caption('Türkçe akademik prototip · Cihazları kontrol etmez · Gerçek bir destek sistemine kayıt göndermez')

@st.cache_resource(show_spinner=False)
def get_router():
    from engine import Router
    return Router(ROOT/'models')

route_tab, results_tab, about_tab = st.tabs(['Talep yönlendir','Ölçülmüş sonuçlar','Nasıl çalışır?'])
with route_tab:
    with st.form('ticket'):
        text=st.text_area('Destek mesajınız',placeholder='Örn. kameram Wi-Fi ağına bağlanmıyor, model CAM-204',height=130,max_chars=3000)
        apply_corrections=st.checkbox('Ürün adlarındaki önerilen yazım düzeltmelerini uygula',value=False)
        submitted=st.form_submit_button('Talebi analiz et',type='primary')
    st.caption('Örnekler: “Garanti belgesini bulamıyorum.” · “Merhaba” · “Yarın hava nasıl?”')
    if submitted:
        if not text.strip():
            st.warning('Önce bir destek mesajı yazın.')
        else:
            info=inspect_text(text)
            with st.expander('İşlenen metin ve ürün bilgileri',expanded=bool(info['corrections'] or info['model_codes'])):
                st.write('Model kodları:',', '.join(info['model_codes']) or 'Bulunamadı')
                st.write('Normalize edilmiş metin:',info['normalized'])
                if info['corrections']:
                    st.write('Yazım önerileri:',info['corrections'])
                    st.caption('Düzeltmeler yalnızca kutuyu işaretlediyseniz model girdisine uygulanır.')
            selected=info['suggested'] if apply_corrections else info['normalized']
            if info['rule_reply']:
                st.success(info['rule_reply'])
                st.caption('Kural tabanlı küçük sohbet yanıtı. Dört model bu mesaj için çalıştırılmadı.')
            else:
                try:
                    with st.spinner('Yerel modeller hazırlanıyor ve mesajınız analiz ediliyor…'):
                        router=get_router()
                        long_input=len(router.tokenizer(selected,add_special_tokens=True)['input_ids'])>64
                        predictions=router.predict_all(selected)
                    primary=predictions[-1]
                    if primary['rejected']:
                        st.warning('Ana yönlendirme: İnsan Temsilci')
                        st.write('INT8 modelinin skoru sabit red eşiğinin altında. Talep insan incelemesine ayrıldı.')
                    else:
                        st.success('Ana yönlendirme: '+primary['route'])
                    st.caption('Ana model: dört katmanlı öğrenci BERT · dinamik INT8 · yerel CPU ('+router.quant_engine+')')
                    if long_input:
                        st.info('Mesaj Transformer için 64 token sınırını aşıyor; bu modeller ilk 64 tokenı kullanıyor.')
                    columns=st.columns(4)
                    for column,p in zip(columns,predictions):
                        with column:
                            with st.container(border=True):
                                st.markdown('**'+p['model']+'**')
                                st.write(p['route'])
                                st.metric('Tahmin skoru',f"%{p['confidence']*100:.2f}")
                                st.caption(f"Red eşiği: %{p['threshold']*100:.4f}")
                                st.caption('Ham kategori: '+p['label'])
                                st.caption(f"Bu çalıştırma: {p['latency_ms']:.2f} ms")
                    st.caption('Skorlar kalibre edilmiş doğruluk garantisi değildir. Canlı süre tek çalıştırmadır; 100 tekrar ortalaması olan rapor tablosundan farklıdır.')
                    st.download_button('Bu analizi indir (JSON)',json.dumps({'input':text,'model_input':selected,'preprocessing':info,'predictions':predictions},ensure_ascii=False,indent=2),file_name='ticket_analysis.json',mime='application/json')
                except Exception as exc:
                    st.error('Model yükleme veya analiz sırasında hata oluştu. Hata ayrıntısını aşağıdan açabilirsiniz.')
                    with st.expander('Hata ayrıntısı'):st.code(str(exc))

with results_tab:
    st.subheader('Kaydedilmiş deney sonuçları')
    path=ROOT/'models/final_test_stage2/final_results.csv'
    if path.exists():
        st.dataframe(pd.read_csv(path),hide_index=True,width='stretch')
        st.download_button('Sonuç tablosunu indir',path.read_bytes(),file_name='final_results.csv',mime='text/csv')
        bench=ROOT/'models/final_test_stage2/all_models_benchmark.json'
        if bench.exists():
            b=json.loads(bench.read_text(encoding='utf-8'))
            st.caption(f"CPU: {b['cpu']} · {b['threads']} iş parçacığı · {b['measured_runs_per_model']} tekrar · ön işleme dahil")
    else:
        st.info('Sonuç tablosu için final_test_stage2.zip arşivini de hazırlayın.')
    st.warning('Bu tablo 120 yapay eğitim mesajı ve 20 yapay test mesajıyla yapılan önceki deneye aittir. Gerçek müşteri verisindeki başarıyı göstermez.')
    st.write('Canlı uygulama regex normalizasyonu ve isteğe bağlı yazım düzeltmesi ekler. Bu ek işlemler eski test tablosunda değerlendirilmedi. Tablo, orijinal deneyin sonuçları olarak korunur; uygulamanın tüm işlem hattına ait yeni başarı iddiası değildir.')

with about_tab:
    st.subheader('Bir mesajın izlediği yol')
    st.markdown('''1. Metnin Unicode ve boşluk düzeni normalize edilir; ürün model kodları regex ile çıkarılır.
2. Ürün adlarına edit distance ile dar kapsamlı yazım önerileri üretilir. Uygulama tercihi kullanıcıdadır.
3. Yalnızca tam selamlaşma veya küçük sohbet mesajları kurallarla yanıtlanır. “Merhaba, cihazım bozuk” model yoluna gider.
4. Dört model aynı seçilmiş metni işler: TF-IDF + Logistic Regression, ortalama Word2Vec + ağ, öğretmen BERT FP32 ve öğrenci BERT INT8.
5. Her model kendi validation eşiğiyle karar verir. INT8 ana modeldir; düşük skorda insan temsilciye ayırır.''')
    st.write('Öğrenci: öğretmenin 0, 3, 7 ve 11. katmanlarıyla başlatılan dört katmanlı BERT. Eğitimde gerçek etiket kaybı ve öğretmen dağılımıyla KL kaybı birlikte kullanıldı. Ardından Linear katmanlarına dinamik INT8 uygulandı.')
    st.caption('Python / PyTorch / Transformers / Gensim / scikit-learn / Streamlit · tüm tahminler CPU’da yerel çalışır.')
