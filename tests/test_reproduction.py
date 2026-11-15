"""Checks for the public reproduction artifact and input installer."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def load_script(name):
    spec=importlib.util.spec_from_file_location(name, ROOT/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReproductionTests(unittest.TestCase):
    def test_bundles_and_pinned_sources_match_manifest(self):
        tool=load_script('reproduce')
        info=tool.manifest()
        tool.checked_archive(info['records'])
        for name,entry in info['profiles'].items():
            with self.subTest(profile=name), zipfile.ZipFile(tool.checked_archive(entry)) as archive:
                self.assertEqual(set(archive.namelist()),set(entry['files']))
                for path,expected in entry['files'].items():
                    self.assertEqual(hashlib.sha256(archive.read(path)).hexdigest(),expected)
                self.assertIn(entry['source_fidelity'],['recorded-runtime-sources','equivalent-settings'])

    def test_workspace_rejects_unrelated_content_and_changed_sources(self):
        tool=load_script('reproduce')
        with tempfile.TemporaryDirectory() as directory:
            work=Path(directory)/'work'
            tool.prepare('feature-power','recorded',work)
            tool.prepare('feature-power','recorded',work)
            (work/'multirocket/multirocket.py').write_text('# altered\n')
            with self.assertRaisesRegex(ValueError,'source changed'):
                tool.prepare('feature-power','recorded',work)
            other=Path(directory)/'other';other.mkdir();(other/'notes.txt').write_text('keep')
            with self.assertRaisesRegex(ValueError,'must be empty'):
                tool.prepare('review','recorded',other)
            self.assertEqual((other/'notes.txt').read_text(),'keep')
            with self.assertRaisesRegex(ValueError,'Unsafe archive path'):
                tool.safe_target(work,'../../escape.py')

    def test_installer_never_overwrites_mismatched_existing_data(self):
        tool=load_script('download_data')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'data.tsv'
            content=b'1\t2\t3\n';expected=hashlib.sha256(content).hexdigest()
            tool.store_verified(path,content,expected)
            tool.store_verified(path,content,expected)
            path.write_bytes(b'other data')
            with self.assertRaisesRegex(ValueError,'Existing file differs'):
                tool.store_verified(path,content,expected)
            self.assertEqual(path.read_bytes(),b'other data')

    def test_ucr_installer_extracts_only_expected_verified_files(self):
        tool=load_script('download_data')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'reproducibility').mkdir()
            content=b'1\t2\t3\n'
            expected={'Coffee/Coffee_TRAIN.tsv':hashlib.sha256(content).hexdigest()}
            with zipfile.ZipFile(root/'reproducibility/records.zip','w') as records:
                records.writestr('reproducibility/inputs.json',json.dumps({'data_hashes':expected}))
            with zipfile.ZipFile(root/'archive.zip','w') as archive:
                archive.writestr('UCRArchive_2018/Coffee/Coffee_TRAIN.tsv',content)
                archive.writestr('../../unwanted.txt',b'extra file')
            tool.ROOT=root
            tool.install_ucr(root/'archive.zip',root/'data')
            self.assertEqual((root/'data/Coffee/Coffee_TRAIN.tsv').read_bytes(),content)
            self.assertEqual(len(list((root/'data').rglob('*.tsv'))),1)
            self.assertFalse((root/'unwanted.txt').exists())
