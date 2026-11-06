import ctypes
import json
import os
import platform
import shutil
from pathlib import Path
class Memory(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong),('total',ctypes.c_ulonglong),
              ('available',ctypes.c_ulonglong),('page_total',ctypes.c_ulonglong),('page_available',ctypes.c_ulonglong),
              ('virtual_total',ctypes.c_ulonglong),('virtual_available',ctypes.c_ulonglong),('extended',ctypes.c_ulonglong)]
data={'platform':platform.platform(),'python':platform.python_version(),'logical_cpus':os.cpu_count(),
      'disk_free_gb':shutil.disk_usage('.').free/1e9}
if os.name=='nt':
    m=Memory();m.length=ctypes.sizeof(m)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        data.update(ram_gib=m.total/1024**3,available_ram_gib=m.available/1024**3)
root=Path(__file__).resolve().parents[1]
(root/'results/hardware.json').write_text(json.dumps(data,indent=2))
print(json.dumps(data,indent=2))
