import torch
import PIL
import os
import pandas as pd
import numpy as np

class RF_Dataset(torch.utils.data.Dataset):

    def __init__(self, csv_path, images_folder, transform = None):
        self.df = pd.read_csv(csv_path)
        #self.df=csv_path
        #print(self.df.iloc[0])
        #print(self.df.iloc[0]["transmitter_address"])
        self.images_folder = images_folder
        self.transform = transform
        #self.class2index = {"cat":0, "dog":1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        filename = self.df.iloc[index]["Burst_name"]
        mac_path= self.df.iloc[index]["transmitter_address"]
        #label = self.class2index[self.df.iloc[index]["transmitter_address"]]
        label=mac_path

        #image = PIL.Image.open(os.path.join(self.images_folder, filename))
        raw_signal = np.fromfile(os.path.join(self.images_folder,mac_path, filename+".bin"), dtype="int16")
        IQ_signal = raw_signal[0::2] + 1j * raw_signal[1::2]
        signal_data=IQ_signal[:1024]
        #if self.transform is not None:
        #    image = self.transform(image)
        return signal_data, label

