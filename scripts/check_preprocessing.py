"""Confirm the decimal-input fix preserves all training/validation source spans."""
import json
import re
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from trialmatch.data import ROOT,read_split,sentences
count=0
for split in ('train','validation'):
    for row in read_split(split):
        old=[(int(i),s.strip()) for i,s in re.findall(r'(?m)^\s*(\d+)\.\s*(.*?)(?=^\s*\d+\.\s|\Z)',row['note'],re.S)]
        assert old==sentences(row['note']),row['annotation_id']
        count+=1
result={'splits_checked':['train','validation'],'rows_checked':count,'source_spans_unchanged':True,
        'note':'Requiring whitespace after a numbered sentence ID fixes decimal-leading free text without changing the previously evaluated validation inputs.'}
(ROOT/'results/preprocessing_regression.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
