"""Double-click entry point: reuse an already running local app safely."""
import json
import runpy
import sys
import urllib.request
import webbrowser
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
URL='http://127.0.0.1:7860'
try:
    with urllib.request.urlopen(URL+'/api/status',timeout=1) as response:
        status=json.load(response)
    if status.get('notice')=='Educational screening prototype. Not a clinical enrollment decision.':
        webbrowser.open(URL)
        raise SystemExit(0)
except (OSError,ValueError):
    pass
runpy.run_path(str(ROOT/'app.py'),run_name='__main__')
