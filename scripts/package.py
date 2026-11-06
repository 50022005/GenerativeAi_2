"""Build a compact submission bundle without base weights, caches, or Git internals."""
import zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
required=['output/pdf/Criterion_Report.pdf','models/adapter/adapter.pt','data/split_manifest.json',
          'results/test_adapted_metrics.json','results/benchmarks.json','results/error_analysis.md']
for f in required:
    if not (ROOT/f).is_file():raise SystemExit('Missing final artifact: '+f)
dest=ROOT/'output/Criterion_Submission.zip'
folders=['trialmatch','scripts','tests','web','report','licenses','models/adapter','data/processed','results','output/pdf']
roots=['README.md','DEMO_GUIDE.md','SUBMISSION_CHECKLIST.md','THIRD_PARTY_NOTICES.md','FINAL_STATUS.md',
       'Setup_CPU.bat','Start_App.bat','Train_CPU.bat','Build_Report.bat','app.py','requirements.txt','pyproject.toml',
       'data/split_manifest.json','data/demo_cases.json','.gitignore','.gitattributes']
files=[ROOT/f for f in roots]
for folder in folders:
    files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file())
with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as out:
    for p in sorted(set(files)):
        rel=p.relative_to(ROOT)
        if any(s in rel.parts for s in ['__pycache__','development']):continue
        if p.name.endswith(('console.log','.aux','.log','.out','.bbl','.blg')):continue
        if str(rel).replace('\\','/')=='report/main.pdf':continue
        out.write(p,rel.as_posix())
print('Submission bundle:',dest,'bytes:',dest.stat().st_size)
