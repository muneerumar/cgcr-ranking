#!/usr/bin/env python3
"""generate booktabs/CSV tables from completed saved results; no inference."""
import argparse
from pathlib import Path
from paper_common import ROOT, RUN, build, write_tables, write_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'paper')
    args = parser.parse_args()
    package = build(ROOT, RUN)
    write_tables(package, args.output)
    write_manifest(package, args.output)
    print(f'Generated {len(package.tables)} CSV/booktabs tables from {RUN}.')


if __name__ == '__main__':
    main()
