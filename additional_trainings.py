import warnings
warnings.filterwarnings("ignore")

import numpy as np
import model_support
from feature_manager import FeatureManager
from keras.models import model_from_json

# CONFIGURATIONS
path_to_datasets = "./Data/Bitfinex/"

epochs = 10
batch_size = 128
look_back = 50
look_forward = 10

pairs_info = {"BTCUSD": "./Data/Bitfinex/BTCUSD/all_in_one.csv",
              "ETHUSD": "./Data/Bitfinex/ETHUSD/all_in_one.csv",
              "LTCUSD": "./Data/Bitfinex/LTCUSD/all_in_one.csv"}

feature_manager = FeatureManager.merge_pairs(pairs_info)

feature_names = ["open", "high", "low", "close"]

(X_train, Y_train), (X_test, Y_test) = \
    feature_manager.load_splitted_datasets(file_path=path_to_datasets,
                                           prefix="bitf_polo_30_",
                                           split_ratio=0.9)
np.random.seed(202)

json_file = open("./Data/StateLESS_LSTM.json", "r")
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)
model.load_weights("./Data/StateLESS_LSTM.h5")

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

model.fit(X_train, Y_train, batch_size=batch_size, verbose=2, epochs=epochs, shuffle=True, validation_split=0.1)

print("\nBuilding prediction and verification...")
verification = model.predict(X_train, batch_size=batch_size)
prediction = model.predict(X_test, batch_size=batch_size)

model_support.save_model(model=model, filename="StateLESS_LSTM", target_dir="./Data/")

# report performance
print("\nDrawing results...")
model_support.plot_prediction(
                title="{} periods look_back".format(40),
                Y_train=Y_train,
                Y_test=Y_test,
                prediction=prediction,
                verification=verification
            )
