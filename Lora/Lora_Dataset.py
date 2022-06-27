import torch
import PIL
import os
import pandas as pd
import numpy as np

class RF_Dataset(torch.utils.data.Dataset):

    def __init__(self, csv_path, images_folder, transform = None):
        self.df = pd.read_csv(csv_path)

        self.images_folder = images_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        filename = self.df.iloc[index]["Burst_name"]
        mac_path= self.df.iloc[index]["transmitter_address"]

        label=mac_path

        raw_signal = np.fromfile(os.path.join(self.images_folder,mac_path, filename+".bin"), dtype="int16")
        IQ_signal = raw_signal[0::2] + 1j * raw_signal[1::2]
        signal_data=IQ_signal[:1024]

        return signal_data, label