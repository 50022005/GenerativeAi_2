"""Combine the unchanged CPU delivery with the verified GPU extension."""
import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    results = args.results.resolve()
    assert results.is_relative_to(ROOT / 'experiments')
    assert (results / 'local_verification.json').is_file()
    additions = [ROOT / 'README.md', ROOT / 'FINAL_STATUS.md', ROOT / 'RUBRIC_REVIEW.md', ROOT / '.gitignore',
                 ROOT / 'KAGGLE_RESULTS.md', ROOT / 'output/tests_all.txt', ROOT / 'output/pdf/Criterion_Kaggle_Study.pdf',
                 ROOT / 'output/kaggle/Criterion_GPU.ipynb', ROOT / 'output/kaggle/upload_manifest.json',
                 ROOT / 'tests/test_kaggle_study.py']
    additions += list((ROOT / 'kaggle_study').glob('*.py')) + [ROOT / 'kaggle_study/README.md']
    additions += [ROOT / 'scripts' / name for name in ('build_kaggle_notebook.py', 'import_kaggle_results.py',
                                                     'report_kaggle.py', 'package_kaggle.py')]
    additions += [p for p in results.rglob('*') if p.is_file()]
    assert all(p.is_file() for p in additions)
    replacements = {p.relative_to(ROOT).as_posix() for p in additions}
    output = ROOT / 'output/Criterion_With_Kaggle.zip'
    with zipfile.ZipFile(ROOT / 'output/Criterion_Submission.zip') as original, zipfile.ZipFile(output, 'w',
                          compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for item in original.infolist():
            if item.filename not in replacements:
                bundle.writestr(item.filename, original.read(item))
        for path in sorted(additions):
            bundle.write(path, path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(output) as bundle:
        assert bundle.testzip() is None
        assert len(bundle.namelist()) == len(set(bundle.namelist()))
    print(output, output.stat().st_size)


if __name__ == '__main__': main()
