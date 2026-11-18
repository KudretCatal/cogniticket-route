"""Offline CPU inference with original label orders and frozen thresholds."""
import hashlib
import json
import sys
import re
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn
from gensim.models import Word2Vec
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAMES = ['TF-IDF', 'Word2Vec + NN', 'BERT öğretmen FP32', 'BERT öğrenci INT8']

def stable_hash(text):
    return int.from_bytes(hashlib.sha256(text.encode('utf-8')).digest()[:8], 'little')

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

class Router:
    def __init__(self, root='models', quant_engine=None):
        self.root = Path(root)
        torch.set_num_threads(2)
        self.baseline_meta = read_json(self.root/'baseline_stage2/metadata.json')
        self.baseline = joblib.load(self.root/'baseline_stage2/baseline.joblib')
        self.teams = self.baseline_meta['teams']
        d = self.root/'word2vec_nn_stage2'
        self.nn_meta = read_json(d/'metadata.json')
        # Gensim stores a reference to the Colab __main__.stable_hash function.
        setattr(sys.modules['__main__'], 'stable_hash', stable_hash)
        self.w2v = Word2Vec.load(str(d/'word2vec.model'))
        self.scaler = joblib.load(d/'scaler.joblib')
        a = self.nn_meta['architecture']
        self.network = nn.Sequential(nn.Linear(a['input_size'],a['hidden_size']), nn.ReLU(), nn.Dropout(a['dropout']), nn.Linear(a['hidden_size'],a['output_size']))
        self.network.load_state_dict(torch.load(d/'neural_network.pt',map_location='cpu',weights_only=True), strict=True)
        self.network.eval()
        d = self.root/'bert_teacher_stage2'
        self.teacher = AutoModelForSequenceClassification.from_pretrained(d, local_files_only=True, attn_implementation='eager').cpu().eval()
        self.teacher_meta = read_json(d/'validation_results.json')
        self.tokenizer = AutoTokenizer.from_pretrained(d, local_files_only=True)
        d = self.root/'bert_student_int8_stage2'
        self.quant_meta = read_json(d/'comparison.json')
        saved_engine = self.quant_meta['runtime']['quantization_engine']
        supported = list(torch.backends.quantized.supported_engines)
        if quant_engine is not None:
            if quant_engine not in supported or quant_engine not in ('x86', 'fbgemm', 'onednn'):
                raise RuntimeError(f'İstenen CPU motoru desteklenmiyor: {quant_engine}; mevcut: {supported}')
            chosen_engine = quant_engine
        else:
            # Windows commonly exposes fbgemm rather than the Linux x86 name.
            chosen_engine = next((name for name in (saved_engine, 'fbgemm', 'x86', 'onednn')
                                  if name in supported and name in ('x86', 'fbgemm', 'onednn')), None)
            if chosen_engine is None:
                raise RuntimeError(f'Uygun INT8 CPU motoru bulunamadı. Mevcut motorlar: {supported}. 64 bit Intel/AMD CPU ve CPU PyTorch kurulumu gerekir.')
        self.quant_engine = chosen_engine
        self.saved_quant_engine = saved_engine
        torch.backends.quantized.engine = chosen_engine
        config = AutoConfig.from_pretrained(d,local_files_only=True)
        config._attn_implementation = 'eager'
        skeleton = AutoModelForSequenceClassification.from_config(config).cpu().eval()
        self.quantized = torch.ao.quantization.quantize_dynamic(skeleton,{nn.Linear},dtype=torch.qint8,inplace=False)
        self.quantized.load_state_dict(torch.load(d/'student_int8_state.pt',map_location='cpu',weights_only=True),strict=True)
        self.quantized.eval()
        self.thresholds = [self.baseline_meta['rejection_threshold'], self.nn_meta['rejection_threshold'], self.teacher_meta['rejection_threshold'], self.quant_meta['same_threshold_for_both']]
        label_orders = [list(self.baseline.classes_), self.nn_meta['labels_in_output_order'], [self.teacher.config.id2label[i] for i in range(self.teacher.config.num_labels)], [self.quantized.config.id2label[i] for i in range(self.quantized.config.num_labels)]]
        self.label_orders = label_orders
        if not all(set(labels)==set(self.teams) for labels in label_orders):
            raise ValueError('Model etiketleri ile ekip listesi uyuşmuyor.')

    def word_features(self,text):
        text=text.replace('İ','i').replace('I','ı').lower()
        tokens=re.findall(r'[a-zçğıöşü0-9]+',text)
        vectors=[self.w2v.wv[t] for t in tokens if t in self.w2v.wv]
        v=np.mean(vectors,axis=0) if vectors else np.zeros(self.w2v.vector_size)
        v=self.scaler.transform(np.asarray([v],dtype=np.float32)).astype(np.float32)
        return torch.from_numpy(v)

    def predict_one(self, index, text):
        start=time.perf_counter()
        with torch.inference_mode():
            if index==0:
                probs=self.baseline.predict_proba([text])[0]
            elif index==1:
                probs=torch.softmax(self.network(self.word_features(text)),dim=-1)[0].numpy()
            else:
                # Include tokenization in each model's live latency, as in final benchmark.
                inputs=self.tokenizer(text,padding='max_length',truncation=True,max_length=64,return_tensors='pt')
                model=self.teacher if index==2 else self.quantized
                probs=torch.softmax(model(**inputs).logits,dim=-1)[0].numpy()
        elapsed=(time.perf_counter()-start)*1000
        position=int(np.argmax(probs)); label=self.label_orders[index][position]
        score=float(probs[position]); threshold=float(self.thresholds[index])
        rejected=score<threshold
        return {'model':MODEL_NAMES[index], 'label':label,'predicted_team':self.teams[label], 'confidence':score,'threshold':threshold,'rejected':rejected,'route':'İnsan Temsilci' if rejected else self.teams[label],'latency_ms':elapsed}

    def predict_all(self,text):
        return [self.predict_one(i,text) for i in range(4)]
