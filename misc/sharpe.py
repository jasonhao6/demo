"""Utility functions for comparison of Sharpe ratio based on daily vs monthly price data.
1. Data loading functions (from local .csv files)
2. Analytic help functions, e.g. normalise
3. Performances stats calculation, e.g. return (CAGR), volatility, Sharpe
"""

import collections
import datetime
import dateutil
import logging
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


# ### ### ### Data loading ### ### ### #

CSV_DIR = os.path.join('D:', 'datasets', 'ETF')
ETF_INFO_PATH = os.path.join('D:', 'datasets', 'tables', 'info_ETF.csv')

COL_DATE = 'Date'
DATE_FORMAT = '%Y/%m/%d'
DAYS_PER_YEAR = 252
MONTHS_PER_YEAR = 12


def get_sector_etf_info(info_path=ETF_INFO_PATH,
                        columns=['Ticker', 'Name'],  # , 'Region', 'Provider'
                        names2drop=['Real Estate', 'Communication Services']):
    """Load Sector ETF ticker and name information """
    info_etf = pd.read_csv(info_path)
    info_sector = info_etf[info_etf['Group'] == 'Sector']
    if names2drop:  # if not []
        info_sector = info_sector[~(info_sector['Name'].isin(names2drop))]
    info_sector = info_sector[columns]
    return info_sector


def get_ts(tickers, item='adjusted close',  start=None, end=None):
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
    if start is None and end is None:
        return ts_df

    # Start and end dates
    if start is not None:
        start = pd.to_datetime(start)
        ts_df = ts_df[start:]
    if end is not None:
        end = pd.to_datetime(end)
        ts_df = ts_df[:end]
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


def plot_normalized_ts(data, title=None, start=None, end=None, rename=None, **kwargs):
    """Plot time-series data in normalized form. """
    df_out = normalise(data)
    if rename is not None:
        df_out = df_out.rename(columns=rename)
    ax = df_out.plot(**kwargs)

    start = df_out.first_valid_index() if start is None else pd.to_datetime(start)
    end = df_out.last_valid_index() if end is None else pd.to_datetime(end)
    date_range = '{} - {}'.format(start.strftime(DATE_FORMAT), end.strftime(DATE_FORMAT))
    if title is not None:
        ax.set_title('{} ({})'.format(title, date_range), fontsize=26)
    ax.legend(fontsize='x-large')
    return ax


def get_year_frac(start_date, end_date):
    """Get number of year in fraction between two dates.
    This implementation is an estimate, revisit here Jason.
    """
    delta = dateutil.relativedelta.relativedelta(end_date, start_date)
    return delta.years + delta.months / 12 + delta.days / 365.25


def compute_cagr(data, start=None, end=None, periods_per_year=None):
    """Compute compounded annual growth rate (CAGR), annualized return.
    Input close price data. """
    first_index = data.first_valid_index()
    last_index = data.last_valid_index()

    start = first_index if start is None else start
    end = last_index if end is None else end

    if periods_per_year is None:
        num_years = get_year_frac(start, end)
    else:
        num_years = len(data) / periods_per_year

    ratio = np.divide(data.loc[last_index], data.loc[first_index])
    cagr = np.power(ratio, 1 / num_years) - 1
    return cagr


def compute_volatility_from_returns(data, periods_per_year=None, start=None, end=None):
    if periods_per_year is None:
        start = data.first_valid_index() if start is None else start
        end = data.last_valid_index() if end is None else end
        num_years = get_year_frac(start, end)
        periods_per_year = len(data.index) / num_years
    return np.sqrt(periods_per_year) * data.std()


def downsample_ts_value(daily, freq='M', keep_first=True):
    df = daily.resample(freq).last()
    if keep_first:
        first_index = daily.first_valid_index()
        if first_index < df.first_valid_index():
            df = pd.concat([daily.loc[[first_index]], df])
    return df


def downsample_price_and_return_data(daily_price):
    """Output (price, return)-tuple data at different frequency ('m', 'd') in form of x[freq] """
    monthly_price = downsample_ts_value(daily_price)
    daily_ret = daily_price.pct_change().dropna(axis=0, how='all')
    monthly_ret = monthly_price.pct_change().dropna(axis=0, how='all')
    return {'d': daily_price, 'm': monthly_price}, {'d': daily_ret, 'm': monthly_ret}


def compute_performance_stats_dm(df_close):
    """Compute basic performance stats return, volatility and Sharpe based on daily and monthly data. """
    # Performance stats based on daily data
    daily_rets = compute_cagr(df_close)
    daily_vols = compute_volatility_from_returns(df_close.pct_change(), periods_per_year=DAYS_PER_YEAR)
    daily_sharpe = daily_rets.divide(daily_vols)

    df_daily = pd.DataFrame({'Return': daily_rets, 'Volatility': daily_vols, 'Sharpe': daily_sharpe})

    # Daily to monthly data
    df_monthly_close = downsample_ts_value(df_close)

    # Performance stats based on monthly data
    monthly_rets = compute_cagr(df_monthly_close)
    monthly_vols = compute_volatility_from_returns(df_monthly_close.pct_change(), periods_per_year=MONTHS_PER_YEAR)
    monthly_sharpe = monthly_rets.divide(monthly_vols)

    df_monthly = pd.DataFrame({'Return': monthly_rets, 'Volatility': monthly_vols, 'Sharpe': monthly_sharpe})

    # Combine daily and monthly for output
    df_stats = pd.concat({'Daily': df_daily, 'Monthly': df_monthly}, axis=1)
    df_stats = df_stats.swaplevel(axis=1).sort_index(axis=1)
    return df_stats


def plot_rolling_volatility(daily_price, num_years=3):
    _, returns = downsample_price_and_return_data(daily_price)

    # rolling volatility estimates from daily returns
    window = num_years * DAYS_PER_YEAR
    daily_rolling_vol = returns['d'].rolling(window=window, min_periods=window - 1).std().dropna(how='all')
    daily_rolling_vol *= np.sqrt(DAYS_PER_YEAR)  # annualization

    # rolling volatility estimates from monthly returns
    window = num_years * MONTHS_PER_YEAR
    monthly_rolling_vol = returns['m'].rolling(window=window, min_periods=window - 1).std().dropna(how='all')
    monthly_rolling_vol *= np.sqrt(MONTHS_PER_YEAR)  # annualization

    # plot volatility estimates daily vs. monthly, ticker by ticker
    tickers = daily_price.columns.tolist()
    fig_map = collections.OrderedDict([(x, None) for x in tickers])
    for tix in tickers:
        title = '{} {}-Year Rolling Volatility Estimates'.format(tix, num_years)
        d_rvol = daily_rolling_vol[tix]
        m_rvol = monthly_rolling_vol[tix]

        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        d_rvol.to_frame('{} Volatility Daily'.format(tix)).plot(ax=ax, linewidth=1)
        m_rvol.to_frame('{} Volatility Monthly'.format(tix)).plot(ax=ax, linestyle=':', linewidth=3)
        ax.set_title(title, fontsize=22)
        ax.legend(fontsize='x-large')
        fig_map[tix] = fig
    return fig_map


# ### ### ### ACF and AR(1) Model ### ### ### #

def plot_acf_dm(daily_close):
    """Plot ACF based on both daily and monthly return data. """
    _, ret_data = downsample_price_and_return_data(daily_close)

    tickers = daily_close.columns.tolist()
    fig_map = collections.OrderedDict([(x, None) for x in tickers])
    for tix in tickers:
        daily_ts = ret_data['d'][tix]
        monthly_ts = ret_data['m'][tix]

        ## ACF plots
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        # fig.suptitle(tix2name.get(tix))

        txt = '{} {} Return Autocorrelation'.format(tix, 'Daily')
        sm.graphics.tsa.plot_acf(daily_ts, lags=14, title=txt, ax=axes[0])

        txt = '{} {} Return Autocorrelation'.format(tix, 'Monthly')
        sm.graphics.tsa.plot_acf(monthly_ts, lags=14, title=txt, ax=axes[1])

        fig_map[tix] = fig
    return fig_map


def run_single_ar_model(ts, p=1):
    """Run AR(p)-model and output relevant parameters. """
    mod = sm.tsa.arima.ARIMA(ts.values, order=(p,0,0))
    res = mod.fit()  # print(res.summary())
    resid = pd.Series(res.resid, index=ts.index)
    ar1_coeff = res.arparams[0]
    return {'ar1_coeff': ar1_coeff, 'resid': resid, 'res': res}


def run_ar_models_dm(daily_close, monthly_close=None, p=1):
    """Run AR(p) model on both daily and monthly return data. """
    if monthly_close is None:
        monthly_close = downsample_ts_value(daily_close)
    daily_returns = daily_close.pct_change().dropna(axis=0, how='all')
    monthly_returns = monthly_close.pct_change().dropna(axis=0, how='all')

    tickers = daily_close.columns.tolist()
    dict_daily = collections.OrderedDict([(tix, None) for tix in tickers])
    dict_monthly = collections.OrderedDict([(tix, None) for tix in tickers])
    for tix in tickers:
        daily_ts = daily_returns[tix]
        arm = run_single_ar_model(daily_ts)
        ar1_coeff = arm['ar1_coeff']
        sigma_vol = np.sqrt(DAYS_PER_YEAR) * arm['resid'].std()
        ar_vol = sigma_vol / (1 - ar1_coeff ** 2)
        s_daily = pd.Series({'AR1_Coeff': ar1_coeff, 'Sigma_Vol': sigma_vol, 'AR_Vol': ar_vol})

        monthly_ts = monthly_returns[tix]
        arm = run_single_ar_model(monthly_ts)
        ar1_coeff = arm['ar1_coeff']
        sigma_vol = np.sqrt(MONTHS_PER_YEAR) * arm.get('resid').std()
        ar_vol = sigma_vol / (1 - ar1_coeff ** 2)
        s_monthly = pd.Series({'AR1_Coeff': ar1_coeff, 'Sigma_Vol': sigma_vol, 'AR_Vol': ar_vol})

        dict_daily[tix] = s_daily
        dict_monthly[tix] = s_monthly
    df_daily = pd.DataFrame(dict_daily).transpose()
    df_monthly = pd.DataFrame(dict_monthly).transpose()
    ar_stats  = pd.concat({'Daily': df_daily, 'Monthly': df_monthly}, axis=1)
    ar_stats = ar_stats .swaplevel(axis=1).sort_index(axis=1)
    return ar_stats


# ### ### ### Randomly Generated i.i.d. data ### ### ### #

def fake_iid_ts_data(n=6, dates=None, periods=10*DAYS_PER_YEAR, seed=101, initial_value=100.):
    """Fake i.i.d. identical and independent time-series data.
    """
    if dates is None:
        dates = pd.date_range(end='2021/01/31', periods=periods, freq='B')
    else:
        periods = len(dates)

    np.random.seed(seed)
    ret_vals = np.random.randn(periods, n)
    random_vol = 0.01 * np.random.randint(11, 25, n)

    tickers = ['Fake {}'.format(x + 1) for x in range(n)]
    s_random_vol = pd.Series(random_vol, index=tickers)

    daily_fake_ret = pd.DataFrame(ret_vals, columns=tickers, index=dates)
    daily_fake_ret = daily_fake_ret.multiply(s_random_vol) / np.sqrt(DAYS_PER_YEAR)
    daily_fake_price = initial_value * (1 + daily_fake_ret).cumprod()

    print(s_random_vol.to_frame('Fake Target Volatility'))
    return daily_fake_price

