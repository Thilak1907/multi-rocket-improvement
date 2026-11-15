#!/usr/bin/env python3
"""Reanalyse recorded experiments or rerun their pinned implementations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'reproducibility'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest():
    return json.loads((ARTIFACTS / 'manifest.json').read_text())


def checked_archive(entry):
    path = ARTIFACTS / entry['archive']
    if digest(path) != entry['sha256']:
        raise ValueError(f'Artifact checksum mismatch: {path}')
    return path


def safe_target(root, name):
    target = (root / name).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f'Unsafe archive path: {name}')
    return target


def write_member(archive, name, root, data_dir):
    target = safe_target(root, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = archive.read(name)
    if name.endswith('.json'):
        data = data.replace(b'${REPO}', str(root).encode())
        value = json.loads(data)
        if isinstance(value, dict) and 'datapath' in value:
            value['datapath'] = str(data_dir)
        data = (json.dumps(value, indent=2) + '\n').encode()
    target.write_bytes(data)


def prepare(profile, mode, workspace=None, data_dir=None):
    info = manifest()
    entry = info['profiles'][profile]
    source = checked_archive(entry)
    records = checked_archive(info['records'])
    work = Path(workspace or ROOT / 'output/reproductions' / mode / profile).resolve()
    data = Path(data_dir or ROOT / 'data/UCRArchive_2018').resolve()
    marker = work / '.reproduction.json'
    identity = dict(profile=profile, mode=mode, source_sha256=entry['sha256'],
                    records_sha256=info['records']['sha256'], data_dir=str(data))
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError('Workspace belongs to another configuration; choose --workspace elsewhere')
        changed = [name for name, expected in entry['files'].items()
                   if not (work / name).is_file() or digest(work / name) != expected]
        if changed:
            raise ValueError('Pinned workspace source changed: ' + ', '.join(changed[:5]))
        return work, entry
    if mode == 'rerun':
        if not data.is_dir():
            raise FileNotFoundError(f'Data not found: {data}; see README data download steps')
        if not (ROOT / 'data/indices109').is_dir():
            raise FileNotFoundError('Run scripts/download_data.py --upstream first')
    if work.exists() and any(work.iterdir()):
        raise ValueError('Workspace must be empty or created by this tool')
    work.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        for name in archive.namelist():
            target = safe_target(work, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    with zipfile.ZipFile(records) as archive:
        if mode == 'recorded':
            names = archive.namelist()
        else:
            # Parent manifests provide split/seed/input hashes, not cached jobs.
            parents = {
                'output/combination_search/20261111/protocol.json',
                'output/review_revision/corrected_v1/protocol.json',
                'output/optimization_comparison/50k_split42_alpha_tie/protocol.json',
            }
            own = f"output/{entry['output']}/protocol.json"
            names = [n for n in archive.namelist() if n.startswith(('data/', 'results/'))
                     or (n in parents and n != own)]
            if profile == 'review':
                # Legacy audit compares fresh predictions with these five records.
                audit_names = ('BME', 'CinCECGTorso', 'FordA', 'SonyAIBORobotSurface1', 'WormsTwoClass')
                names += [f'output/combination_search/20261111_fast30/confirmation/{d}__fast30__0.json'
                          for d in audit_names]
        for name in names:
            write_member(archive, name, work, data)
    if mode == 'rerun':
        (work / 'data/UCRArchive_2018').symlink_to(data, target_is_directory=True)
        indices = ROOT / 'data/indices109'
        (work / 'data/indices109').symlink_to(indices, target_is_directory=True)
    marker.write_text(json.dumps(identity, indent=2) + '\n')
    return work, entry


def execute(work, module, *arguments):
    env = dict(os.environ, VECLIB_MAXIMUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='4', NUMBA_NUM_THREADS='4',
               MPLCONFIGDIR=str(work / '.matplotlib'))
    print('Running:', sys.executable, '-m', module, *map(str, arguments), flush=True)
    subprocess.run([sys.executable, '-m', module, *map(str, arguments)], cwd=work, env=env, check=True)


def summarize(work, profile, entry):
    output = 'output/' + entry['output']
    report = 'reports/' + entry['report']
    if profile == 'review':
        execute(work, 'experiments.summarize_revision', '--output', output, '--report-dir', report)
    elif profile in ('accuracy', 'feature-power', 'ridge-range'):
        module = {'accuracy': 'accuracy', 'feature-power': 'feature_power', 'ridge-range': 'ridge_range'}[profile]
        execute(work, 'experiments.summarize_' + module, '--output', output, '--report', report)
    elif profile == 'legacy-combinations':
        execute(work, 'experiments.summarize_combinations',
                '--output', 'output/combination_search/20261111_repeats3',
                '--refinement-output', 'output/combination_search/20261111_fast30', '--report-dir', report)
    else:
        execute(work, 'experiments.summarize_optimizations', '--output', output, '--report-dir', report)
    print('Report:', work / report / 'report.md')


def rerun(work, profile, entry, smoke=False):
    if smoke:
        # One dataset exercises the pinned numerical code without a full benchmark.
        shutil.copy2(ROOT / 'main_ucr.py', work / 'main_ucr.py')
        execute(work, 'main_ucr', '-d', 'data/UCRArchive_2018', '-p', 'ArrowHead',
                '-n', '30000', '-t', '4', '--classifier', 'fast_ridge', '--memory-efficient',
                '--prediction-batch-size', '256', '--optimized-transform', '--fused-prediction',
                '--seed', '20261111')
        print('Smoke test only; no archive-level result was computed.')
        return
    if profile == 'review':
        execute(work, 'experiments.review_revision', '--stage', 'all')
    elif profile in ('accuracy', 'feature-power', 'ridge-range'):
        module = {'accuracy': 'accuracy_search', 'feature-power': 'feature_power_search',
                  'ridge-range': 'ridge_range_search'}[profile]
        execute(work, 'experiments.' + module)
    elif profile == 'legacy-combinations':
        execute(work, 'experiments.search_combinations', '--stage', 'all')
        execute(work, 'experiments.repeat_confirmation')
        execute(work, 'experiments.refine_fast30', '--stage', 'all')
        execute(work, 'experiments.repeat_confirmation', '--stage', 'stress')
        execute(work, 'experiments.refine_fast30', '--stage', 'stress')
    else:
        execute(work, 'experiments.compare_optimizations', '--output', 'output/' + entry['output'],
                '--repeats', '3' if profile == 'legacy-initial' else '1', '--stage', 'all')
    summarize(work, profile, entry)


def verify_inputs(data_dir):
    info = manifest()
    with zipfile.ZipFile(checked_archive(info['records'])) as archive:
        inputs = json.loads(archive.read('reproducibility/inputs.json'))
    errors = []
    for name, expected in inputs['data_hashes'].items():
        path = data_dir / name
        if not path.is_file() or digest(path) != expected:
            errors.append(str(path))
    for name, expected in inputs['index_hashes'].items():
        path = ROOT / 'data/indices109' / (name + '_INDICES_TRAIN.txt')
        if not path.is_file() or digest(path) != expected:
            errors.append(str(path))
    if errors:
        raise ValueError('Missing or changed inputs:\n' + '\n'.join(errors[:20]))
    print('Verified 218 TSV files and 109 TRAIN-index files against the recorded experiment.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('list', 'verify', 'summarize', 'run'))
    parser.add_argument('profile', nargs='?', choices=tuple(manifest()['profiles']) + ('all',))
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/UCRArchive_2018')
    parser.add_argument('--workspace', type=Path)
    parser.add_argument('--smoke', action='store_true', help='Run ArrowHead only; not a benchmark result')
    parser.add_argument('--allow-equivalent', action='store_true',
                        help='Allow early legacy settings whose exact original source was not preserved')
    args = parser.parse_args()
    if args.action == 'list':
        for name, entry in manifest()['profiles'].items():
            print(f"{name:22} reports/{entry['report']}  [{entry['source_fidelity']}]")
        return
    if args.action == 'verify':
        for entry in [manifest()['records'], *manifest()['profiles'].values()]:
            checked_archive(entry)
        verify_inputs(args.data_dir.resolve())
        return
    if not args.profile:
        parser.error('Choose an experiment profile')
    if args.smoke and (args.action != 'run' or args.profile != 'review'):
        parser.error('--smoke is supported by run review only')
    if args.profile == 'all' and (args.workspace or args.action == 'run'):
        parser.error('Use separate profiles/workspaces for training; all is supported for summarize only')
    profiles = manifest()['profiles'] if args.profile == 'all' else [args.profile]
    for profile in profiles:
        entry = manifest()['profiles'][profile]
        if args.action == 'run' and entry['source_fidelity'] != 'recorded-runtime-sources' and not args.allow_equivalent:
            parser.error('Exact early source was not preserved. Use --allow-equivalent for a settings rerun, or summarize for recorded results.')
        mode = 'recorded' if args.action == 'summarize' else 'rerun'
        work, entry = prepare(profile, mode, args.workspace, args.data_dir)
        if args.action == 'summarize':
            summarize(work, profile, entry)
        else:
            rerun(work, profile, entry, args.smoke)


if __name__ == '__main__':
    main()
