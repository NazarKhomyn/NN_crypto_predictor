import pandas as pd
import numpy as np

"""
    Indexes: ["MACD", "CCI", "ATR", "EMA20", "MA5", "MA10", "MTM6", "MTM12", "ROC", "SMI", "WVAD]
"""


# MACD
def MACD(df, n_fast, n_slow):
    EMAfast = pd.Series(pd.ewma(df['close'], span=n_fast, min_periods=n_slow - 1))
    EMAslow = pd.Series(pd.ewma(df['close'], span=n_slow, min_periods=n_slow - 1))
    MACD_index = pd.Series(EMAfast - EMAslow, name='MACD')
    df = df.join(MACD_index)
    return df


# Commodity Channel Index
def CCI(df, n):
    PP = (df['high'] + df['low'] + df['close']) / 3
    CCI_index = pd.Series((PP - pd.rolling_mean(PP, n)) / pd.rolling_std(PP, n), name='CCI')
    df = df.join(CCI_index)
    return df


# Average True Range
def ATR(df, n):
    i = 0
    TR_l = [0]
    while i < df.index[-1]:
        TR = max(df.get_value(i + 1, 'high'), df.get_value(i, 'close')) - min(df.get_value(i + 1, 'low'), df.get_value(i, 'close'))
        TR_l.append(TR)
        i = i + 1
    TR_s = pd.Series(TR_l)
    ATR_index = pd.Series(pd.ewma(TR_s, span=n, min_periods=n), name='ATR')
    df = df.join(ATR_index)
    return df


# Bollinger Bands
def BOLL(df, n):
    MA_index = pd.Series(pd.rolling_mean(df['close'], n))
    MSD = pd.Series(pd.rolling_std(df['close'], n))
    b1 = 4 * MSD / MA_index
    B1 = pd.Series(b1, name='BOLL')
    df = df.join(B1)
    b2 = (df['close'] - MA_index + 2 * MSD) / (4 * MSD)
    B2 = pd.Series(b2, name='BOOL' + str(n))
    df = df.join(B2)
    return df


# Exponential Moving Average
def EMA(df, n):
    EMA_index = pd.Series(pd.ewma(df['close'], span=n, min_periods=n - 1), name='EMA' + str(n))
    df = df.join(EMA_index)
    return df


# Moving Average
def MA(df, n):
    MA_index = pd.Series(pd.rolling_mean(df['close'], n), name='MA' + str(n))
    df = df.join(MA_index)
    return df


# Momentum
def MTM(df, n):
    M = pd.Series(df['close'].diff(n), name='MTM' + str(n))
    df = df.join(M)
    return df


# Rate of Change
def ROC(df, n):
    M = df['close'].diff(n - 1)
    N = df['close'].shift(n - 1)
    ROC_index = pd.Series(M / N, name='ROC')
    df = df.join(ROC_index)
    return df


# Stochastic Momentum Index
def SMI(df, r, s):
    M = pd.Series(df['close'].diff(1))
    aM = abs(M)
    EMA1 = pd.Series(pd.ewma(M, span=r, min_periods=r-1))
    aEMA1 = pd.Series(pd.ewma(aM, span=r, min_periods=r-1))
    EMA2 = pd.Series(pd.ewma(EMA1, span=s, min_periods=s-1))
    aEMA2 = pd.Series(pd.ewma(aEMA1, span=s, min_periods=s-1))
    SMI_index = pd.Series(EMA2 / aEMA2, name='SMI')
    df = df.join(SMI_index)
    return df


# Williams Accumulation Distribution
def WVAD(df, n):
    ad = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low']) * df['total_volume']
    M = ad.diff(n - 1)
    N = ad.shift(n - 1)
    ROC_index = M / N
    WVAD_index = pd.Series(ROC_index, name='WVAD')
    df = df.join(WVAD_index)
    return df


def add_indicators(df):
    df = MACD(df, 12, 26)
    df = CCI(df, 20)
    df = ATR(df, 14)
    df = BOLL(df, 20)
    df = EMA(df, 20)
    df = MA(df, 5)
    df = MA(df, 10)
    df = MTM(df, 6)
    df = MTM(df, 12)
    df = ROC(df, 12)
    df = SMI(df, 25, 13)
    df = WVAD(df, 28)

    return df


"""
def typical_price(df):
    # Typical Price.
    # Formula:
    # TPt = (HIGHt + LOWt + CLOSEt) / 3

    return (df["high"] + df["close"] + df["low"]) / 3


def money_flow(df):
    # Money Flow.
    # Formula:
    # MF = VOLUME * TYPICAL PRICE

    return df["total_volume"] * typical_price(df)


def money_flow_index(df, period):
    # Money Flow Index.
    # Formula:
    # MFI = 100 - (100 / (1 + PMF / NMF))

    mf = money_flow(df)
    tp = typical_price(df)

    flow = [tp[i] > tp[i-1] for i in range(1, tp.shape[0])]

    pf = [mf[i] if flow[i] else 0 for i in range(len(flow))]
    nf = [mf[i] if not flow[i] else 0 for i in range(len(flow))]

    pmf = np.array([sum(pf[i+1-period: i+1]) for i in range(period - 1, len(pf))])
    nmf = np.array([sum(nf[i+1-period: i+1]) for i in range(period - 1, len(nf))])

    # Dividing by 0 is not an issue, it turns the value into NaN which we would
    # want in that case
    # with warnings.catch_warnings():
    #    warnings.simplefilter("ignore", category=RuntimeWarning)
    # mfi = fill_for_noncomputable_vals(close_data, mfi)

    money_ratio = np.divide(pmf, nmf)
    mfi = 100 - (100 / (1 + money_ratio))

    return mfi


def on_balance_volume(df):
    # On Balance total_volume.
    # Formula:
    # start = 1
    # if CLOSE[t] > CLOSE[t-1]
    #    obv[t] = obv[t-1] + volume[t]
    # elif CLOSE[t] < CLOSE[t-1]
    #    obv[t] = obv[t-1] - volume[t]
    # elif CLOSE[t] == CLOST[t-1]
    #     obv[t] = obv[t-1]

    volume = df["total_volume"]
    close = df["close"]

    obv = np.zeros(df.shape[0])
    obv[0] = 1

    for i in range(1, df.shape[0]):
        if close[i] > close[i - 1]:

            obv[i] = obv[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - volume[i]

        elif close[i] == close[i - 1]:
            obv[i] = obv[i - 1]

    return obv


def chaikin_index(df):
    ad = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low']) * df['total_volume']

    return df.join(pd.Series(pd.ewma(ad, span=3, min_periods=2) - pd.ewma(ad, span=10, min_periods=9), name='Chaikin'))


def add_indicators(trades_df):
    temp = money_flow_index(trades_df, 14)

    mfi = np.zeros(14)
    mfi = np.concatenate((mfi, temp))

    trades_df['OBV'] = on_balance_volume(trades_df)
    trades_df['MFI'] = mfi
    trades_df = chaikin_index(trades_df)

    return trades_df

"""