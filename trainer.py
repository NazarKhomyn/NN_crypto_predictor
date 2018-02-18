import warnings
warnings.filterwarnings("ignore")

import numpy as np
import model_support
from feature_manager import FeatureManager

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import configparser


np.random.seed(202)
CONFIG = configparser.ConfigParser()


class Trainer:
    def __init__(self, feature_manager, prefix, path_to_datasets, input_columns, output_column,
                 split_ratio=0.8, look_forward=10, look_back=40, is_saved_dataset=False):
        self.feature_manager = feature_manager
        self.model = Sequential()

        self.history = None

        self.look_back = look_back
        self.look_forward = look_forward

        self.input_columns = input_columns
        self.output_column = output_column
        self.feature_manager = feature_manager

        if not is_saved_dataset:
            (self.X_train, self.Y_train), (self.X_test, self.Y_test) = \
                self.feature_manager.split_dataset(look_back=look_back,
                                                   look_forward=look_forward,
                                                   split_ratio=split_ratio,
                                                   save_to=path_to_datasets,
                                                   prefix=prefix,
                                                   input_columns=input_columns,
                                                   output_column=output_column)
        else:
            (self.X_train, self.Y_train), (self.X_test, self.Y_test) = \
                feature_manager.load_splitted_datasets(file_path=path_to_datasets,
                                                       prefix=prefix,
                                                       split_ratio=split_ratio)
        CONFIG["Dataset info"] = {"split_ratio": split_ratio,
                                  "look_forward": look_forward,
                                  "look_back": look_back}

        CONFIG["Shapes"] = {"X_train": str(self.X_train.shape),
                            "Y_train": str(self.Y_train.shape),
                            "X_test": str(self.X_test.shape),
                            "Y_test": str(self.Y_test.shape)}

    def run(self, epochs=6, batch_size=128, validation_split=0.1):
        amount_of_features = len(self.input_columns)
        amount_of_outputs = len(self.output_column)

        print("\n"
              "------ DATASETS ------\n"
              "Train set shape: {}\n"
              "Train target set shape: {}\n"
              "Test set shape: {}\n"
              "Test target set shape: {}\n"
              "\n"
              "Amount of features: {}\n".format(self.X_train.shape, self.Y_train.shape, self.X_test.shape,
                                                self.Y_test.shape, amount_of_features))

        # stacked autoencoder

        self.model.add(LSTM(amount_of_features, kernel_initializer="random_uniform", return_sequences=True,
                       input_shape=(self.look_back, amount_of_features)))
        self.model.add(Dropout(0.1))

        self.model.add(LSTM(120, kernel_initializer="random_uniform", return_sequences=True))
        self.model.add(Dropout(0.1))

        self.model.add(LSTM(120, kernel_initializer="random_uniform", return_sequences=True))
        self.model.add(Dropout(0.1))

        self.model.add(LSTM(120, kernel_initializer="random_uniform", return_sequences=True))
        self.model.add(Dropout(0.1))

        self.model.add(LSTM(amount_of_features, kernel_initializer="random_uniform", return_sequences=True))
        self.model.add(Dropout(0.1))

        # LSTM layer
        self.model.add(LSTM(480, kernel_initializer="random_uniform"))
        self.model.add(Dropout(0.25))

        self.model.add(Dense(amount_of_outputs, activation="linear", kernel_initializer="random_uniform"))

        # Compile for regression
        self.model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        self.history = self.model.fit(self.X_train, self.Y_train, batch_size=batch_size,
                                      verbose=2, epochs=epochs, shuffle=True, validation_split=validation_split)

        CONFIG["Train info"] = {"epochs": epochs,
                                "validation_split": validation_split,
                                "batch_size": batch_size}

    def save_model(self, filename="StateLESS_LSTM", target_dir="./Data/"):
        print("\nSave {} model".format(filename))
        model_support.save_model(model=self.model, filename=filename, target_dir=target_dir)

    def build_prediction(self, batch_size=128):
        CONFIG["Prediction info"] = {"batch_size": batch_size}

        print("\nBuilding prediction and verification...")
        verification = self.model.predict(self.X_train, batch_size=batch_size)
        prediction = self.model.predict(self.X_test, batch_size=batch_size)

        verification = self.feature_manager.rescale(verification, columns=self.output_column)
        prediction = self.feature_manager.rescale(prediction, columns=self.output_column)

        return prediction, verification

    def get_report(self, prediction, verification):
        # report performance
        np.save("prediction.npy", prediction)
        print("\nDrawing results...")

        model_support.plot_prediction(
            title=self.input_columns + ["{} periods look_back".format(self.look_back)],
            Y_train=self.feature_manager.rescale(self.Y_train, columns=self.output_column),
            Y_test=self.feature_manager.rescale(self.Y_test, columns=self.output_column),
            prediction=prediction,
            verification=verification,
            names=self.output_column
        )

        model_support.plot_train_info(self.history)
        model_support.pack(names=self.output_column)


def main(output_column):
    # pairs_info = {"BTCUSD": "./Data/Bitfinex/BTCUSD/all_in_one.csv",
    #               "ETHUSD": "./Data/Bitfinex/ETHUSD/all_in_one.csv",
    #               "LTCUSD": "./Data/Bitfinex/LTCUSD/all_in_one.csv",
    #               "XRPUSD": "./Data/Bitfinex/XRPUSD/all_in_one.csv",
    #               "BCHUSD": "./Data/Bitfinex/BCHUSD/all_in_one.csv",
    #               "ETCUSD": "./Data/Bitfinex/ETCUSD/all_in_one.csv",
    #               "ZECUSD": "./Data/Bitfinex/ZECUSD/all_in_one.csv",
    #               "XMRUSD": "./Data/Bitfinex/XMRUSD/all_in_one.csv",
    #               "NEOUSD": "./Data/Bitfinex/NEOUSD/all_in_one.csv",
    #
    #               "BTCETH": "./Data/Poloniex/BTCETH/all_in_one.csv",
    #               "BTCXRP": "./Data/Poloniex/BTCXRP/all_in_one.csv"}

    # feature_manager = FeatureManager.merge_pairs(pairs_info)

    # feature_manager.add_total_market_cap(path="./Data/cmc_5s.csv")

    feature_manager = FeatureManager()
    feature_manager.read_dataset_from_csv("./Data/BTC-ETH-ETC-LTC-XRP-XMR-BCH-ZEC-NEO.csv")

    # feature_manager.save_dataset_to_csv("./Data/BTC-ETH-ETC-LTC-XRP-XMR-BCH-ZEC-NEO.csv")

    trades_features = [
        "open",
        "high",
        "low",
        "close",
        "total_volume",
        "total_amount",
    ]
    trading_indicators = [
        'MACD',
        'CCI',
        'ATR',
        'BOLL',
        'BOOL20',
        'EMA20',
        'MA5',
        'MA10',
        'MTM6',
        'MTM12',
        'ROC',
        'SMI',
        'WVAD'
    ]

    all_features = []

    for pair in [
        # Bitfinex
        "BTCUSD", "BCHUSD", "ETCUSD",
        "ETHUSD", "LTCUSD", "NEOUSD",
        "XMRUSD", "XRPUSD", "ZECUSD",
        # Poloniex
        "BTCETH", "BTCXRP"
    ]:
        for trades_feature in trades_features:
            all_features.append(pair + "_" + trades_feature)
        for trading_indicator in trading_indicators:
            all_features.append(pair + "_" + trading_indicator)

    # all_features.append("market_cap")

    feature_manager.extract_features(all_features, fill_nan=True, scale=True)

    # CONFIGURATIONS
    path_to_datasets = "./Data/Bitfinex/"
    input_columns = []

    for pair in [
        # Bitfinex
        "BTCUSD", "BCHUSD", "ETCUSD",
        "ETHUSD", "LTCUSD", "NEOUSD",
        "XMRUSD", "XRPUSD", "ZECUSD",
        # Poloniex
        "BTCETH", "BTCXRP"
    ]:
        for trades_feature in trades_features:
            input_columns.append(pair + "_" + trades_feature)
        for trading_indicator in trading_indicators:
            input_columns.append(pair + "_" + trading_indicator)

    # input_columns.append("market_cap")

    prefix = "couple_30_"

    trainer = Trainer(feature_manager=feature_manager,
                      prefix=prefix,
                      path_to_datasets=path_to_datasets,
                      output_column=output_column,
                      input_columns=input_columns,
                      is_saved_dataset=False)

    trainer.run()
    trainer.save_model()

    prediction, verification = trainer.build_prediction()

    with open('config.yaml', 'w') as configfile:
        CONFIG.write(configfile)

    trainer.get_report(prediction=prediction, verification=verification)


for output in [["XRPUSD_close", "LTCUSD_close"],
               ["BTCUSD_close"],
               ["ETHUSD_close"],
               ["XRPUSD_close"],
               ["ETHUSD_close", "BTCUSD_close", "XRPUSD_close", "LTCUSD_close"]]:
    main(output_column=output)