"""Import the five existing archives; never downloads or executes archive code."""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ARCHIVES = {
    'baseline_stage2.zip': 'baseline_stage2',
    'word2vec_nn_stage2.zip': 'word2vec_nn_stage2',
    'bert_teacher_stage2.zip': 'bert_teacher_stage2',
    'bert_student_int8_stage2.zip': 'bert_student_int8_stage2',
    'final_test_stage2.zip': 'final_test_stage2',
}

def prepare(source, target):
    source, target = Path(source), Path(target)
    # Validate all archives before making changes.
    manifest_path = Path(__file__).resolve().parent/'archive_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for filename, root in ARCHIVES.items():
        path = source / filename
        if not path.is_file():
            raise FileNotFoundError(f'Eksik dosya: {path}')
        if filename in manifest:
            h = hashlib.sha256()
            with path.open('rb') as src:
                for block in iter(lambda: src.read(1024*1024), b''):
                    h.update(block)
            if h.hexdigest() != manifest[filename]['sha256']:
                raise ValueError(f'Arşiv içeriği doğrulama kaydıyla uyuşmuyor: {filename}')
        with zipfile.ZipFile(path) as archive:
            for entry in archive.infolist():
                p = Path(entry.filename)
                if p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0] != root:
                    raise ValueError(f'Beklenmeyen ZIP yolu: {entry.filename}')
    for filename in ARCHIVES:
        with zipfile.ZipFile(source / filename) as archive:
            for entry in archive.infolist():
                output = target / entry.filename
                if entry.is_dir():
                    output.mkdir(parents=True, exist_ok=True)
                else:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(entry) as src, output.open('wb') as dst:
                        shutil.copyfileobj(src, dst)
    print('Beş arşiv hazırlandı. Modeller yeniden eğitilmedi.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archives', default='archives')
    parser.add_argument('--output', default='models')
    args = parser.parse_args()
    prepare(args.archives, args.output)
