"""Utility functions for Hierarchical risk parity demo.
1. Data loading functions (from local .csv files)
2. Analytic help functions, e.g. normalise
"""

import collections
import datetime
import dateutil
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ### ### ### Data loading ### ### ### #

CSV_DIR = os.path.join('D:', 'datasets', 'ETF')
ETF_INFO_PATH = os.path.join('D:', 'datasets', 'tables', 'info_ETF.csv')
COL_DATE = 'Date'
DATE_FORMAT = '%Y/%m/%d'


def get_sector_etf_info(info_path=ETF_INFO_PATH,
                        columns=['Ticker', 'Name'],  # , 'Region', 'Provider'
                        names2drop=['Real Estate', 'Communication Services']):
    info_etf = pd.read_csv(info_path)
    info_sector = info_etf[info_etf['Group'] == 'Sector']
    if names2drop:  # if not []
        info_sector = info_sector[~(info_sector['Name'].isin(names2drop))]
    info_sector = info_sector[columns]
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


def get_sector_etf_close(tickers, benchmark='SPY', start=None, end=None, item='adjusted close'):
    # Sector ETF adjusted close
    df_close = get_ts(tickers, item=item)
    df_start = df_close.first_valid_index()
    df_end = df_close.last_valid_index()

    # Start and end dates
    start = df_start if start is None else pd.to_datetime(start)
    end = df_end if end is None else pd.to_datetime(end)
    df_close = df_close[start:end]

    # benchmark
    if benchmark is None:
        bench_close = None
    else:
        benchmark = 'SPY'
        bench_close = get_ts(benchmark, item=item)
        bench_close = bench_close[benchmark]
        bench_close = bench_close[start:end]
    return df_close, bench_close


def get_monthly_price_return(price, bench=None, start=None, end=None):
    df_start = price.first_valid_index()
    df_end = price.last_valid_index()
    start = df_start if start is None else pd.to_datetime(start)
    end = df_end if end is None else pd.to_datetime(end)
    price = price[start:end]
    price = price[start:end].resample('M').last()
    mon_ret = price.pct_change().dropna(how='all')

    if bench is None:
        return mon_ret
    else:
        bench = bench[start:end].resample('M').last()
        mon_ben_ret = bench.pct_change().dropna()
        out_ex_ret = mon_ret.subtract(mon_ben_ret, axis=0)
        return out_ex_ret


# ### ### ### Analytic Help Functions  ### ### ### #

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


def get_year_frac(start_date, end_date):
    """Get number of year in fraction between two dates.
    This implementation is an estimate, revisit here Jason.
    """
    delta = dateutil.relativedelta.relativedelta(end_date, start_date)
    return delta.years + delta.months / 12 + delta.days / 365.25


def compute_volatility_from_returns(data, periods_per_year=None, start=None, end=None):
    if periods_per_year is None:
        start = data.first_valid_index() if start is None else start
        end = data.last_valid_index() if end is None else end
        num_years = get_year_frac(start, end)
        periods_per_year = len(data.index) / num_years
    return np.sqrt(periods_per_year) * data.std()


def compute_cagr(data, start=None, end=None):
    """Input close price data. """
    start = data.first_valid_index() if start is None else start
    end = data.last_valid_index() if end is None else end
    num_years = get_year_frac(start, end)

    first_index = data.first_valid_index()
    last_index = data.last_valid_index()
    ratio = np.divide(data.loc[last_index], data.loc[first_index])
    cagr = np.power(ratio, 1 / num_years) - 1
    return cagr


def get_vol_return_pair(price, bench=None, name=None, weight=None, cluster=None):
    rets = compute_cagr(price)
    vols = compute_volatility_from_returns(price.pct_change())

    if name is None:
        df = pd.DataFrame({'Volatility': vols, 'Return': rets})
    else:
        df = pd.DataFrame({'Name': name, 'Volatility': vols, 'Return': rets})
    if weight is not None:
        df['Weight'] = weight
    if cluster is not None:
        df['Cluster'] = cluster
    return df
