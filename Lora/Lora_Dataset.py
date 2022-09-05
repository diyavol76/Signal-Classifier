import torch
import PIL
import os
import pandas as pd
import numpy as np
from sklearn import preprocessing


class RF_Dataset(torch.utils.data.Dataset):

    def __init__(self, csv_path, images_folder, transform = None,target_transform=None):
        self.df = pd.read_csv(csv_path)

        self.data_folder = images_folder
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        self.df.dropna()
        labels=self.df['transmitter_address'].unique()
        print("labels",labels)
        di = {"Dev_1": 1, "Dev_2": 2}
        self.df["transmitter_address"].map(di)
        #self.df['transmitter_address'].replace({"Dev_2": 2})


        print("labels", labels)
        filename = self.df.iloc[index]["Burst_name"]
        mac_path= self.df.iloc[index]["transmitter_address"]
        antenna = self.df.iloc[index]["Antenna"]

        label=mac_path

        raw_signal = np.fromfile(os.path.join(self.data_folder, mac_path, filename + ".bin"), dtype="int16")
        signal_data=raw_signal
        #signal_data = signal_data.astype(np.float)
        signal_data=torch.from_numpy(signal_data)
        label = torch.from_numpy(label)
        #IQ_signal = raw_signal[0::2] + 1j * raw_signal[1::2]
        #signal_data=IQ_signal[:1024]

        if self.transform :
            signal_data = self.transform(signal_data)
        if self.target_transform:
            label = self.target_transform(label)

        return signal_data, label