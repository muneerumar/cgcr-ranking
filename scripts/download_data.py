#!/usr/bin/env python3
"""Print the pinned data plan; download and validate only with --download."""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import shutil
from prepare_data import fetch_ranges

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(1024*1024):result.update(block)
    return result.hexdigest()


def download(item,root=ROOT):
    root=Path(root).resolve();path=(root/item['path']).resolve()
    if not path.is_relative_to(root/'data/raw'):
        raise ValueError('Data destinations must be beneath data/raw')
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():fetch_ranges(item,logging.getLogger('cgcr.data'),root,workers=4)
    if path.stat().st_size!=item['content_length'] or digest(path)!=item['sha256']:
        raise ValueError('Data integrity mismatch: '+path.name)
    return path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download',action='store_true')
    args=parser.parse_args(argv)
    files=json.loads((ROOT/'configs/data_sources.json').read_text())['files']
    plan=dict(files=files,total_bytes=sum(f['content_length'] for f in files),
              free_bytes=shutil.disk_usage(ROOT).free,network_requested=args.download)
    print(json.dumps(plan,indent=2))
    if not args.download:return 0
    if plan['total_bytes']>60_000_000_000:raise ValueError('Plan exceeds 60 GB')
    # Downloads are retained here; preparation may release compressed files later.
    remaining=sum(f['content_length'] for f in files if not (ROOT/f['path']).exists())
    if remaining+2_500_000_000>plan['free_bytes']:
        raise OSError('Insufficient free space for retained downloads and reserve')
    for item in files:
        print('Verified '+str(download(item)),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
