import os
import pickle
import pandas as pd


def read_pickle(filename, columns=None):
    dataset = []

    with open(filename, "br") as dataset_file:
        while True:
            try:
                data = pickle.load(dataset_file)
                if data:
                    dataset.append(data)
                else:
                    break
            except EOFError:
                break

    return pd.DataFrame(dataset, columns=columns)


def read_sampled(target_dir):
    """
    Gather info together in one data frame.

    :param target_dir: dir with sampled csv filed
    :return: united data frame with all data
    """
    names = os.listdir(target_dir)
    all_df = []

    names.sort()

    for name in names:
        all_df.append(pd.read_csv(target_dir + name, sep=","))

    trades_df = all_df[0]
    for i in range(1, len(all_df)):
        trades_df = trades_df.append(all_df[i], ignore_index=True)

    return trades_df
