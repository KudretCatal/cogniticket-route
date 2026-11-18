"""Reload/inference regression check against frozen reported predictions.
This does not train, tune or replace the held-out results.
"""
import json
from pathlib import Path
import pandas as pd
from engine import Router

ROOT = Path(__file__).resolve().parent

def main():
    router = Router(ROOT/'models')
    reference = pd.read_csv(ROOT/'models/final_test_stage2/test_predictions.csv')
    mismatches=[]
    raw_label_differences=[]
    count=0
    for row in reference.to_dict('records'):
        if row['model']=='BERT öğrenci FP32':
            continue
        index=['TF-IDF','Word2Vec + NN','BERT öğretmen FP32','BERT öğrenci INT8'].index(row['model'])
        actual=router.predict_one(index,row['text'])
        expected_reject=row['final_route']=='human_agent'
        count+=1
        if actual['label']!=row['predicted_label_before_rejection']:
            raw_label_differences.append({'model':row['model'],'id':row['id'],'rejected':actual['rejected']})
        if actual['rejected']!=expected_reject or (not expected_reject and actual['label']!=row['predicted_label_before_rejection']):
            mismatches.append({'model':row['model'],'id':row['id'],'expected_label':row['predicted_label_before_rejection'],'actual_label':actual['label'],'expected_reject':expected_reject,'actual_reject':actual['rejected']})
    result={'comparisons':count,'mismatches':mismatches,'raw_label_differences':raw_label_differences,'note':'Raw archived inputs; no new preprocessing or threshold changes.'}
    (ROOT/'verification_local.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if mismatches:raise SystemExit(1)

if __name__=='__main__':main()
