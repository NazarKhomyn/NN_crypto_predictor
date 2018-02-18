import numpy as np
from data_sampler import DataSampler

dir_with_pickles = "./Data/Bitfinex/BTC-USD/"
save_to = "./Data/Bitfinex/Processed/"

timedelta = np.timedelta64(5, "s")

DataSampler.sample_files_from_dir(dir_with_pickles, save_to, timedelta)
