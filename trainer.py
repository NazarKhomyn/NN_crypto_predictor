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
                 split_ratio=0.8, look_forward=10, look_back=70, is_saved_dataset=False):
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

    def run(self, epochs=11, batch_size=128, validation_split=0.1):
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
        self.model.add(Dropout(0.2))

        self.model.add(LSTM(120, kernel_initializer="random_uniform", return_sequences=True))
        #self.model.add(Dropout(0.2))

        self.model.add(LSTM(120, kernel_initializer="random_uniform", return_sequences=True))
        #self.model.add(Dropout(0.2))

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

        train_set = np.array([self.X_train[i] for i in range(0, self.X_train.shape[0],
                                                             int(0.50 * self.look_back))])

        train_target_set = np.array([self.Y_train[i] for i in range(0, self.Y_train.shape[0],
                                                                    int(0.50 * self.look_back))])

        self.history = self.model.fit(train_set, train_target_set, batch_size=batch_size,
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
            verification = verification,
            #prediction=self.feature_manager.rescale(prediction, columns=self.output_column),
            #verification=self.feature_manager.rescale(verification, columns=self.output_column),
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
    feature_manager.read_dataset_from_csv("BTC_data_extremas.csv")

    # feature_manager.save_dataset_to_csv("./Data/BTC-ETH-ETC-LTC-XRP-XMR-BCH-ZEC-NEO.csv")

    trades_features = [
        "BTCUSD_open",
        "BTCUSD_high",
        "BTCUSD_low",
        "BTCUSD_close",
        "BTCUSD_total_volume",
        "BTCUSD_total_amount",
    ]

    extrema_features = [
        'extremas',
        'dists_to_extrema',
        'growth_decrease'
    ]

    all_features = trades_features + extrema_features

    feature_manager.extract_features(all_features, fill_nan=True, scale=True)

    # CONFIGURATIONS
    path_to_datasets = "./Data/"

   # input_columns = trades_features + ['extremas', 'growth_decrease']
    input_columns = all_features
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


main(output_column=["dists_to_extrema"])
