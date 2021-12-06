"""Utility functions for Hierarchical risk parity demo.
1. Regular help functions, e.g. normalise
2. Data loading functions (from local .csv files)
"""

import collections
import datetime
import dateutil
import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


# ### ### ### Regular Help Functions  ### ### ### #

def normalise(data, initial_value=100.):
    if isinstance(data, pd.Series):
        first_index = data.first_valid_index()
        if first_index is None:
            return data
        return data / data[first_index] * initial_value
    else:  # DataFrame
        out = data.copy()
        for c in data.columns:
            out[c] = normalise(data[c], initial_value)
        return out

# ### ### ### Data loading ### ### ### #

CSV_DIR = os.path.join('D:', 'datasets', 'ETF')
ETF_INFO_PATH = os.path.join('D:', 'datasets', 'tables', 'info_ETF.csv')
COL_DATE = 'Date'
DATE_FORMAT = '%Y/%m/%d'


def get_sector_etf_info(info_path=ETF_INFO_PATH, names2drop=['Real Estate']):
    info_etf = pd.read_csv(info_path)
    info_sector = info_etf[info_etf['Group'] == 'Sector']
    if names2drop:  # if not []
        info_sector = info_sector[~(info_sector['Name'].isin(names2drop))]
    return info_sector


def get_ts(tickers, item='adjusted close'):
    """Load time-series data for a list of input tickers.
     Input items can be 'adjusted close', 'close', 'volume', 'dividend amount'.
     """
    if isinstance(tickers, str):
        return get_ts([tickers], item=item)

    ts_dict = collections.OrderedDict([(x, None) for x in tickers])
    for tix in tickers:
        df = get_single_tsdf(ticker=tix)
        if df is not None:
            if item in df:
                ts_dict[tix] = df[item]
            else:
                logger.warning('Ticker {} column {} NOT FOUND'.format(tix, item))
    ts_df = pd.concat(ts_dict, axis=1)
    return ts_df


def get_single_tsdf(ticker, csv_dir=CSV_DIR, col_date=COL_DATE):
    """Load a time-series table for a single ticker, with all columns.
    It searches all data in different csv directories. """
    if not isinstance(ticker, str):
        logger.error('load_single_tsdf(name) accepts only single ticker, but type(name) = {}'.format(type(ticker)))

    csv_name = '{}.csv'.format(ticker)
    csv_path = os.path.join(csv_dir, csv_name)
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path, index_col=col_date, parse_dates=[col_date])  # check index
        logger.debug('Load time-series {} from {}'.format(ticker, csv_path))
        return df

    logger.warning('Ticker {} time-series : NOT FOUND'.format(ticker))
    return None



