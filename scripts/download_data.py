#!/usr/bin/env python3
"""Download UCR inputs and the pinned upstream resample indices."""
import argparse
import getpass
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
UCR_URL = 'https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download(url, target):
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + '.part')
    request = urllib.request.Request(url, headers={'User-Agent': 'MultiRocket-reproduction'})
    print('Downloading', url, flush=True)
    try:
        with urllib.request.urlopen(request, timeout=120) as source, partial.open('wb') as dest:
            shutil.copyfileobj(source, dest)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target


def store_verified(target, content, expected):
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError(f'Downloaded file differs from recorded input: {target}')
    if target.exists():
        if digest(target) != expected:
            raise ValueError(f'Existing file differs; move it aside before retrying: {target}')
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)


def install_upstream(archive_path, destination):
    reference = json.loads((ROOT / 'reproducibility/upstream_files.json').read_text())
    if digest(archive_path) != reference['archive_sha256']:
        raise ValueError('Pinned upstream archive checksum mismatch')
    with zipfile.ZipFile(archive_path) as archive:
        prefix = archive.namelist()[0].split('/')[0] + '/'
        for name, expected in reference['files'].items():
            store_verified(destination / name, archive.read(prefix + name), expected)
    print(f"Installed/verified {len(reference['files'])} metadata, index, and published-result files.")


def install_ucr(archive_path, destination, password=None):
    with zipfile.ZipFile(ROOT / 'reproducibility/records.zip') as records:
        expected = json.loads(records.read('reproducibility/inputs.json'))['data_hashes']
    with zipfile.ZipFile(archive_path) as archive:
        selected = {}
        for info in archive.infolist():
            parts = info.filename.replace('\\', '/').split('/')
            name = '/'.join(parts[-2:])
            if name in expected:
                selected[name] = info
        if set(selected) != set(expected):
            raise ValueError('Archive is missing expected TSVs: ' + ', '.join(sorted(set(expected) - set(selected))[:10]))
        if any(info.flag_bits & 1 for info in selected.values()) and password is None:
            password = getpass.getpass('UCR archive password (see the official briefing document): ')
        key = password.encode() if password else None
        for name, info in selected.items():
            # Whitelisted relative names come from our input manifest, not the ZIP.
            store_verified(destination / name, archive.read(info, pwd=key), expected[name])
    datasets = len({name.split('/')[0] for name in expected})
    print(f'Installed/verified {len(expected)} TSV files for {datasets} datasets.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', action='store_true', help='Download pinned metadata, indices, and paper results')
    parser.add_argument('--ucr', action='store_true', help='Download the official UCR ZIP and extract the 109 datasets')
    parser.add_argument('--upstream-archive', type=Path, help='Use an already downloaded pinned upstream ZIP')
    parser.add_argument('--ucr-archive', type=Path, help='Use an already downloaded official UCR ZIP')
    parser.add_argument('--destination', type=Path, default=ROOT, help='Repository root receiving data/ and results/')
    parser.add_argument('--password', help='Optional UCR ZIP password; otherwise prompt when encrypted')
    args = parser.parse_args()
    if not any((args.upstream, args.ucr, args.upstream_archive, args.ucr_archive)):
        parser.error('Choose --upstream, --ucr, or a local archive')
    destination = args.destination.resolve()
    cache = destination / '.cache/downloads'
    if args.upstream or args.upstream_archive:
        ref = json.loads((ROOT / 'reproducibility/upstream_files.json').read_text())
        archive = args.upstream_archive or download(
            'https://codeload.github.com/ChangWeiTan/MultiRocket/zip/' + ref['revision'], cache / 'upstream.zip')
        install_upstream(archive, destination)
    if args.ucr or args.ucr_archive:
        archive = args.ucr_archive or download(UCR_URL, cache / 'UCRArchive_2018.zip')
        install_ucr(archive, destination / 'data/UCRArchive_2018', args.password)


if __name__ == '__main__':
    main()
