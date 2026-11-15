# Reproduction artifacts

These files make the reported experiments independent of a developer's ignored
`output/` directory.

- `records.zip`: recorded jobs, prediction arrays, small dataset metadata, and
  published reference CSVs; no raw UCR series or environment files.
- `*-source.zip`: pinned Python sources for an experiment. Analysis uses the
  retained bug-fixed summary scripts; runtime sources are checked against the
  original manifests where available.
- `manifest.json`: bundle/file SHA-256 values, output mappings, original protocol
  hashes, and explicit exceptions for two unsnapshotted early implementations.
- `upstream_files.json`: the pinned upstream revision and checksums for downloading
  archive indices, metadata, and author result CSVs.
- `environment.json`: the measured Python/numeric-package versions and thread counts.

Run `python scripts/reproduce.py summarize all` from the repository root to
rebuild reports without training. See [the reproduction guide](../docs/reproducing.md)
for new runs, data preparation, and the source-fidelity limitations.

The archived source retains [GPL-3.0](../LICENSE) and the
[HYDRA attribution](../experiments/vendor/README.md). Original absolute paths in
record JSON are made portable; original protocol hashes are retained separately.
