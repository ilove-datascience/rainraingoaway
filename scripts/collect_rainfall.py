"""Run rainfall archiving without starting Telegram or model inference."""
import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from scraping.rainfall import collect_missing_days, main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.once:
        failed = collect_missing_days()
        print('Rainfall archive pass complete; failed days:', failed, flush=True)
        sys.exit(bool(failed))
    main()
