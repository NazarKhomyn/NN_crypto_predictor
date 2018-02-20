import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import mpld3

from matplotlib.gridspec import GridSpec
from keras.models import model_from_json
import tarfile
import time
import datetime
import os


def save_model(model, filename, target_dir=None):
    if target_dir is None:
        target_dir = "./Models/Basic/"

    model_json = model.to_json()

    # write model to json file
    json_file = open("{}{}.json".format(target_dir, filename), "w")
    json_file.write(model_json)
    json_file.close()

    # save weights
    model.save_weights("{}{}.h5".format(target_dir, filename))


def read_model(filename, target_dir=""):
    print("Read {} model".format(filename))

    json_file = open("{}{}.json".format(filename, target_dir), "r")
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("{}{}.h5".format(filename, target_dir))

    return loaded_model


def plot_prediction(title, Y_train, Y_test, verification, prediction, names):
    prediction = prediction.T
    verification = verification.T

    Y_train = Y_train.T
    Y_test = Y_test.T

    for i in range(prediction.shape[0]):
        fig = plt.figure(figsize=(17, 8))

        gs = GridSpec(2, 1)

        ax1 = plt.subplot(gs[0, :])
        ax1.set_title("Test set", fontsize=30)
        ax1.plot(Y_test[i], label="ACTUAL", color="b")
        ax1.plot(prediction[i], label="PREDICTION", color="r")

        plt.xticks(rotation=90)
        plt.legend()
        plt.grid()

        ax2 = plt.subplot(gs[1, :])
        ax2.set_title("Train set", fontsize=30)
        ax2.plot(Y_train[i], label="ACTUAL", color="b")
        ax2.plot(verification[i], label="VERIFICATION", color="r")

        plt.xticks(rotation=90)
        plt.legend()
        plt.grid()

        fig.suptitle(str(title), fontsize=35)

        plt.show()

        mpld3.save_html(fig, '{}_output.html'.format(names[i]))

        plt.close()


def plot_train_info(history_info):
    history = history_info.history

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 8))
    try:
        plt.subplots_adjust(left=0.045, right=0.99, top=0.92, bottom=0.1)

        ax1.plot(history["val_loss"], c="r", label="Value Loss")
        plt.legend()

        ax2.plot(history["val_mean_absolute_error"], c="b", label="Value MAE")
        plt.legend()

        ax3.plot(history["loss"], c="r", label="Loss")
        plt.legend()

        ax4.plot(history["val_mean_absolute_error"], c="b", label="MAE")
        plt.legend()

    except KeyError:
        print("Lol")

    mpld3.save_html(fig, "fit_history.html")
    plt.close()


def pack(names):
    for i in range(len(names)):
        names[i] = "{}_output.html".format(names[i])

    names.append("fit_history.html")
    names.append("config.yaml")

    tar_name = "{}.tar".format(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%dT%s'))
    print("\nPacking reports to " + tar_name)

    tar = tarfile.open(tar_name, "w")

    for name in names:
        tar.add(name)

    tar.add("./Data/StateLESS_LSTM.json")
    tar.add("./Data/StateLESS_LSTM.h5")

    tar.close()

    for name in names:
        os.remove(name)
