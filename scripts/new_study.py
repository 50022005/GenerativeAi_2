"""Archive original evidence before a deliberate fresh reproduction."""
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
archive=(ROOT/'archive'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')).resolve()
if ROOT not in archive.parents:
    raise RuntimeError('Archive path must remain in the project.')
archive.mkdir(parents=True,exist_ok=False)
for source,relative in [(ROOT/'results','results'),(ROOT/'models/adapter','adapter')]:
    source=source.resolve()
    if ROOT not in source.parents:
        raise RuntimeError('Source path must remain in the project.')
    if source.exists(): source.rename(archive/relative)
(ROOT/'results').mkdir(exist_ok=True)
print('Original results and adapter archived at',archive)
print('Run the README experiment sequence in order. The old paper remains until analysis/compilation regenerates it.')
