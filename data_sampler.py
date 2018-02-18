import numpy as np
import pandas as pd

import data_reader as reader
import os


class DataSampler:
    def __init__(self, df, time_delta, start, end):
        """
        :param df: dataframe with all data
        :param time_delta: time_delta for sampling
        :param start: start sampling time
        :param end: end sampling time
        """
        self.df = df
        self.time_delta = time_delta

        self.start_times = np.arange(start=start,
                                     stop=end,
                                     step=time_delta,
                                     dtype="datetime64[ms]")

        self.amount_of_breakdowns = self.start_times.shape[0]

        # must have
        self.opens = np.zeros(self.amount_of_breakdowns)
        self.closes = np.zeros(self.amount_of_breakdowns)
        self.highs = np.zeros(self.amount_of_breakdowns)
        self.lows = np.zeros(self.amount_of_breakdowns)

        # optional data
        self.total_amount = np.zeros(self.amount_of_breakdowns)
        self.total_volume = np.zeros(self.amount_of_breakdowns)

        self.total_rate_of_purchases = np.zeros(self.amount_of_breakdowns)
        self.total_rate_of_sales = np.zeros(self.amount_of_breakdowns)

        # some statistics
        self.average_sale_rate = np.zeros(self.amount_of_breakdowns)
        self.average_sale_amount = np.zeros(self.amount_of_breakdowns)
        self.average_buy_rate = np.zeros(self.amount_of_breakdowns)
        self.average_buy_amount = np.zeros(self.amount_of_breakdowns)

    def _get_partial_df(self, start_date):
        partial_df = self.df[(self.df.date >= start_date)
                             & (self.df.date < start_date + self.time_delta)]

        return partial_df

    def _get_single_sample_item(self, prev_day_close, index_of_current_period, axis=0):
        partial_df = self._get_partial_df(start_date=self.start_times[index_of_current_period])

        if partial_df.shape[0] != 0:
            # if we have not empty time period

            first = partial_df[partial_df.date == self.start_times[index_of_current_period]]

            if first.shape[0] != 0:
                # have trades in the beginning of time period
                self.opens[index_of_current_period] = first.rate.values[0]

            else:
                # haven't trades in the beginning of time period,
                # open rate defines as close rate of previous one
                self.opens[index_of_current_period] = partial_df.rate.values[0]

            self.closes[index_of_current_period] = partial_df.rate.values[-1]
            self.lows[index_of_current_period] = partial_df.rate.values.min(axis=axis)
            self.highs[index_of_current_period] = partial_df.rate.values.max(axis=axis)

            # select info about sales during mentioned period
            sales = partial_df[partial_df.amount < 0]

            # select info about purchases during mentioned period
            purchases = partial_df[partial_df.amount > 0]

            if purchases.shape[0] != 0:
                self.total_rate_of_purchases[index_of_current_period] = np.sum(np.abs(purchases.total.values))

            if sales.shape[0] != 0:
                self.total_rate_of_sales[index_of_current_period] = np.sum(np.abs(sales.total.values))

            self.total_volume[index_of_current_period] = np.sum(np.abs(partial_df.total.values))
            self.total_amount[index_of_current_period] = np.sum(np.abs(partial_df.amount.values))

            # some statistics
            if sales.shape[0] != 0:
                self.average_sale_rate[index_of_current_period] = np.abs(sales.rate.values).mean()
                self.average_sale_amount[index_of_current_period] = np.abs(sales.amount.values.mean())

            if purchases.shape[0] != 0:
                self.average_buy_rate[index_of_current_period] = np.abs(purchases.rate.values).mean()
                self.average_buy_amount[index_of_current_period] = np.abs(purchases.rate.values).mean()

        else:
            # if we have empty time period
            if index_of_current_period != 0:
                self.closes[index_of_current_period] = self.closes[index_of_current_period - 1]
                self.lows[index_of_current_period] = self.closes[index_of_current_period - 1]
                self.highs[index_of_current_period] = self.closes[index_of_current_period - 1]
                self.opens[index_of_current_period] = self.closes[index_of_current_period - 1]
            elif prev_day_close != 0:
                # there is empty time period, but we have data from previous day
                self.closes[index_of_current_period] = prev_day_close
                self.lows[index_of_current_period] = prev_day_close
                self.highs[index_of_current_period] = prev_day_close
                self.opens[index_of_current_period] = prev_day_close
            else:
                # we haven't had anything :c
                print("Oops. You have first period {} without any trades. "
                      "Values set as zero.".format(self.start_times[index_of_current_period]))

                self.closes[index_of_current_period] = 0
                self.lows[index_of_current_period] = 0
                self.highs[index_of_current_period] = 0
                self.opens[index_of_current_period] = 0

    def generate_sample_items(self, prev_day_close):
        """
        Method for generation daily sample items.

        :param prev_day_close: previous day close rate in case empty first time sample
        :return: pandas data frame with sampled data
        """
        for i in np.arange(self.amount_of_breakdowns):
            self._get_single_sample_item(prev_day_close=prev_day_close, index_of_current_period=i)

        summary = pd.DataFrame(data={"start_time": self.start_times,
                                     "open": self.opens,
                                     "close": self.closes,
                                     "high": self.highs,
                                     "low": self.lows,
                                     "total_volume": self.total_volume,
                                     "total_amount": self.total_amount,
                                     "total_rate_of_purchases": self.total_rate_of_purchases,
                                     "total_rate_of_sales": self.total_rate_of_sales,
                                     "average_buy_rate": self.average_buy_rate,
                                     "average_buy_amount": self.average_buy_amount,
                                     "average_sale_rate": self.average_buy_rate,
                                     "average_sale_amount": self.average_buy_amount},

                               columns=["start_time",
                                        "open",
                                        "high",
                                        "low",
                                        "close",
                                        "total_volume",
                                        "total_amount",
                                        "total_rate_of_purchases",
                                        "total_rate_of_sales",
                                        "average_buy_rate",
                                        "average_sale_rate",
                                        "average_buy_amount",
                                        "average_sale_amount"])

        enriched_df = DataSampler._add_additional_data(summary)

        return enriched_df

    @staticmethod
    def _add_additional_data(market_info):
        """
        Adding additional data to market_info DataFrame:
            1) close_off_high = 2 * (high - close) / (high - low) - 1
            2) volatility = (high - low) / open

        :param market_info: pd.DataFrame with columns ['open', 'high', 'low', 'close'] at least
        :return updated market_info
        """

        kwargs = {
            'close_off_high': lambda x: 2 * (x['high'] - x['close']) / (x['high'] - x['low']) - 1,
            'volatility': lambda x: (x['high'] - x['low']) / (x['open'])
        }

        market_info = market_info.assign(**kwargs)

        return market_info

    @staticmethod
    def sample_files_from_dir(target_dir, save_to_dir, time_delta,
                              file_type="pickle"):
        """
        Read all pickle files from target_dir. Then convert each of them in sampled csv file and save to save_to_dir.
        Sampling is conducted with fixed timedelta.

        :param target_dir: directory with necessary pickle files
        :param save_to_dir: reformatted data in csv format will been written to this file
        :param file_type: type of file with data
        :param start_date: start of parsing
        :param end_date: end of parsing
        :param time_delta: time period for data sampling
        :param name: name
        """

        # just little "crutch" for saving previous day close value
        prev_day_close = 0

        for filename in np.sort(os.listdir(target_dir)):
            print("\nRead file " + filename)
            if file_type == "pickle":
                trades_df = reader.read_pickle(target_dir + filename, columns=["ID", "date", "amount", "rate"])
                trades_df["date"] = np.array(trades_df.date, dtype="datetime64[ms]")

            elif file_type == "feather":
                trades_df = pd.read_feather("{}{}"
                                            .format(target_dir, filename))[["date", "rate", "type", "amount", "total"]]
                trades_df["date"] = np.array(trades_df["date"], dtype="datetime64[ms]")
            else:
                trades_df = pd.read_csv("{}{}"
                                        .format(target_dir, filename))

            rates = trades_df["rate"].copy()
            amount = trades_df["amount"].copy()

            trades_df["total"] = pd.Series(amount * rates, index=trades_df.index)

            # print(trades_df.head())

            start_date = np.datetime64("{}T00:00:00".format(filename[:10]))

            end_date = np.datetime64("{}T23:59:59".format(filename[:10]))

            sampler = DataSampler(df=trades_df, time_delta=time_delta, start=start_date, end=end_date)

            summary = sampler.generate_sample_items(prev_day_close=prev_day_close)

            prev_day_close = summary["close"].values[-1]

            name = filename[:-7]

            # print(summary[["start_time", "open", "high", "low", "close", "total_volume"]].head())

            print("Save to: {}{}.csv".format(save_to_dir, name))

            summary.to_csv("{}{}.csv".format(save_to_dir, name), index=False)
