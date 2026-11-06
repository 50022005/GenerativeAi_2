"""Read tensor header metadata; no additional model copy is loaded into RAM."""
import json
import math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
with (ROOT/'models/base/model.safetensors').open('rb') as f:
    n=int.from_bytes(f.read(8),'little')
    header=json.loads(f.read(n))
count=0;draft=0
for key,entry in header.items():
    if key=='__metadata__':continue
    size=math.prod(entry['shape']);count+=size
    if not key.startswith('model.layers.') or int(key.split('.')[2])<6:
        draft+=size
result={'base_parameters':count,'draft_parameters':draft,'adapter_parameters':92160,
        'target_fp32_weight_mib':count*4/1024**2,'draft_fp32_weight_mib':draft*4/1024**2,
        'target_kv_bytes_per_token_batch1':30*2*3*64*4,
        'draft_definition':'First 6/30 blocks with original embeddings and final normalization; uncalibrated; same tokenizer.',
        'alternative_360m_parameters_from_config':361821120,
        'alternative_360m_fp32_weight_mib':361821120*4/1024**2}
(ROOT/'results/model_inventory.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
