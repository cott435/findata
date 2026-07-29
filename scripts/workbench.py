"""Launch the analysis workbench (4-tab Panel app).

Run from the project root:
    python -m scripts.workbench [--universe-n 150 | --tickers AAPL MSFT ...]
        [--start 2013-01-01] [--port 5006] [--db-path data/stock.db]

The universe defaults to the deterministic coverage-filtered sample from
findata.sample_universe; cross-sectional tabs (Compare IC, PCA, Market) use
it wholesale, the Stock tab picks single names from it.
"""

import argparse
import logging

from findata.configs import setup_logging
from findata.database.db_manager import DBManager

logger = logging.getLogger('scripts.workbench')


def main():
    parser = argparse.ArgumentParser(description='Serve the analysis workbench.')
    parser.add_argument('--tickers', nargs='+', default=None,
                        help='Explicit universe (default: sample_universe).')
    parser.add_argument('--universe-n', type=int, default=150)
    parser.add_argument('--start', default='2013-01-01')
    parser.add_argument('--port', type=int, default=5006)
    parser.add_argument('--db-path', default=None)
    parser.add_argument('--no-show', action='store_true',
                        help="Don't open a browser window.")
    args = parser.parse_args()
    setup_logging('workbench.log')

    from findata import sample_universe
    from findata.analysis.plotting.workbench import WorkbenchApp, WorkbenchData

    db = DBManager(args.db_path)
    universe = ([t.upper() for t in args.tickers] if args.tickers
                else sample_universe(n=args.universe_n, db_path=args.db_path))
    logger.info('Workbench universe: %d tickers, start=%s', len(universe), args.start)

    data = WorkbenchData(db, universe, start=args.start)
    WorkbenchApp(data).serve(port=args.port, show=not args.no_show)


if __name__ == '__main__':
    main()
