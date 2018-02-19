import numpy as np
import pandas as pd
import data_reader
import indicators
from scipy import signal


class FeatureManager:
    def __init__(self, trades_df=None):
        self.trades_df = trades_df
        self.features = None

        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None

        self.scale_params = None

        self.X = None
        self.Y = None

    def set_trades_df(self, df):
        self.trades_df = df

    def read_dataset_from_csv(self, file_path):
        print("\nReading trades data frame from csv...")

        self.trades_df = pd.read_csv(file_path, engine="python", index_col=0)

    def save_dataset_to_csv(self, file_path):
        self.trades_df.to_csv(file_path)

    def read_csv_from_dir(self, target_dir):
        self.trades_df = data_reader.read_sampled(target_dir=target_dir)

    @staticmethod
    def scale(data):
        average = np.mean(data, axis=0)
        standard_deviation = np.std(data, axis=0)

        scale_info = {"avg": average, "std": standard_deviation}

        return (data - average) / standard_deviation, scale_info

    def rescale(self, data, columns):
        temp = data.T

        for i in range(len(columns)):
            average = self.scale_params[columns[i]]["avg"]
            standard_deviation = self.scale_params[columns[i]]["std"]

            temp[i] = temp[i] * standard_deviation + average

        return temp.T

    def load_splitted_datasets(self, file_path, prefix, split_ratio):
        print("\nLoading datasets...")

        X = np.load("{}{}X.npy".format(file_path, prefix))
        Y = np.load("{}{}Y.npy".format(file_path, prefix))

        dataset_size = X.shape[0]
        train_set_size = int(dataset_size * split_ratio)

        self.X_train = X[:train_set_size]
        self.Y_train = Y[:train_set_size]

        self.X_test = X[train_set_size:]
        self.Y_test = Y[train_set_size:]

        return (self.X_train, self.Y_train), (self.X_test, self.Y_test)

    def split_dataset(self, look_back, look_forward, input_columns, output_column, prefix,
                      save_to=None, split_ratio=None, train_set_size=None, dataset_size=None,
                      shuffle=False):

        if self.features is not None:
            if dataset_size is None:
                dataset_size = self.trades_df.shape[0]

            if train_set_size is None:
                train_set_size = int(dataset_size * split_ratio)

            input_df = self.features[input_columns]
            output_df = self.features[output_column]

            input_df.index = np.array(input_df.index, dtype="int32")
            output_df.index = np.array(input_df.index, dtype="int32")

            print("\nExtracting set...")
            X = np.array([input_df[(input_df.index > i - look_back) & (input_df.index <= i)].values
                          for i in np.arange(look_back, dataset_size - look_forward)], dtype="float32")

            # X_test = np.array([input_df[(input_df.index > i - look_back) & (input_df.index <= i)].values
            #                    for i in np.arange(train_set_size, dataset_size - look_forward)], dtype="float32")

            print("Extracting target set...")
            Y = np.array([output_df[output_df.index == i + look_forward].values[0]
                          for i in np.arange(look_back, dataset_size - look_forward)], dtype="float32")

            # Y_test = np.array([output_df[output_df.index == i + look_forward].values[0]
            #                    for i in np.arange(train_set_size, dataset_size - look_forward)], dtype="float32")

            self.X = X
            self.Y = Y

            if shuffle:
                np.random.shuffle(X)
                np.random.shuffle(Y)

            self.X_train = X[:train_set_size]
            self.Y_train = Y[:train_set_size]

            self.X_test = X[train_set_size:]
            self.Y_test = Y[train_set_size:]

            if save_to is not None:
                print("\nSaving sets...")
                np.save("{}{}X.npy".format(save_to, prefix), X)
                np.save("{}{}Y.npy".format(save_to, prefix), Y)

            return (self.X_train, self.Y_train), (self.X_test, self.Y_test)

        else:
            print("You have to extract features before.")

    def add_order_book(self, filename):
        data = pd.read_csv(filename)

        columns = ["min_ask",
                   "max_bid",
                   "0.1% ask amount",
                   "0.2% ask amount",
                   "0.5% ask amount",
                   "0.1% ask count",
                   "0.2% ask count",
                   "0.5% ask count",
                   "0.1% bid amount",
                   "0.2% bid amount",
                   "0.5% bid amount",
                   "0.1% bid count",
                   "0.2% bid count",
                   "0.5% bid count"]

        for column in columns:
            self.trades_df[column] = data[column]

    def add_indicators(self):
        print("\nATTENTION! Don't forget to smooth your data before!\n")
        self.trades_df = indicators.add_indicators(self.trades_df)

    def extract_features(self, columns, scale=False, fill_nan=False, filtered=False):
        if filtered:
            self.smooth_data_from_columns(columns=columns)

        features = self.trades_df[columns]

        scale_params = {}

        if fill_nan:
            features = features.fillna(0)

        if scale:
            for column in columns:
                features[column], scale_info = FeatureManager.scale(features[column])
                scale_params[column] = scale_info

            self.scale_params = scale_params

        if fill_nan:
            features = features.fillna(0)

        self.features = features

        print("Assert null: {}".format(features.isnull().values.any()))

        return features

    def get_trades_df(self):
        return self.trades_df

    def merge_with_trades_df(self, df, prefix, on):
        feature_names = df.columns.values.tolist()

        column_name_replacer = {}

        for feature_name in feature_names:
            if feature_name == "start_time":
                column_name_replacer["start_time"] = "start_time"
            else:
                column_name_replacer[feature_name] = prefix + "_" + feature_name

        df = df.rename(index=str, columns=column_name_replacer)

        if self.trades_df is None:
            self.trades_df = df
        else:
            self.trades_df = pd.merge(self.trades_df, df, on=on)

    @staticmethod
    def merge_pairs(cryptos):
        pairs = list(cryptos.keys())
        global_features = FeatureManager()

        for pair in pairs:
            df = pd.read_csv(cryptos[pair], engine="python")

            try:
                global_features.merge_with_trades_df(df, on="start_time", prefix=pair)
            except Exception:
                print("Unable " + pair)

        return global_features

    @staticmethod
    def smooth_data(data, antialiasind_factor=5):
        win = signal.hann(antialiasind_factor)
        filtered = signal.convolve(data, win, mode='same') / sum(win)

        for i in range(antialiasind_factor):
            filtered[i] = data[i]
            filtered[-i] = data[-i]

        return filtered

    def smooth_data_from_columns(self, columns):
        for column in columns:
            self.trades_df[column] = FeatureManager.smooth_data(self.trades_df[column].values)

    def add_total_market_cap(self, path):
        df = pd.read_csv(path, names=["start_time", "market_cap"], engine="python")
        df.start_time = np.array(df.start_time, dtype="datetime64[s]")

        new_df = df[(df.start_time >= np.datetime64("2018-01-21T00:00:00"))
                    & (df.start_time < np.datetime64("2018-02-06T14:21:50"))]

        self.trades_df["market_cap"] = new_df["market_cap"]

        return new_df["market_cap"]
