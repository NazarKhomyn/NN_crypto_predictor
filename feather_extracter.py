import numpy as np
from data_sampler import DataSampler
from feature_manager import FeatureManager

basic_path = "./Data/Poloniex/"


for pair in ["BTCETH", "BTCGAS", "BTCXRP"]:
    print("Extracting " + pair)
    DataSampler.sample_files_from_dir(target_dir=basic_path + pair + "/",
                                      save_to_dir=basic_path + pair + "/",
                                      time_delta=np.timedelta64(30, "s"),
                                      file_type="csv",
                                      start_date=np.datetime64("2018-01-21T00:00:00"),
                                      end_date=np.datetime64("2018-02-06T14:21:00"),
                                      name=pair+"_30")

for pair in ["BTCETH", "BTCGAS", "BTCXRP"]:
    feature_manager = FeatureManager()
    feature_manager.read_dataset_from_csv("{}{}/{}_30.csv".format(basic_path, pair, pair))
    print(feature_manager.get_trades_df().head())

    feature_manager.add_indicators()
    feature_manager.save_dataset_to_csv("{}{}/all_in_one_30.csv".format(basic_path, pair))
