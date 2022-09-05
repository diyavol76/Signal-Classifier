import torch
import PIL
import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
import random
from torch import nn
import h5py


class RF_Dataset(torch.utils.data.Dataset):

    def __init__(self, csv_path, images_folder, transform = None,target_transform=None):
        csv = pd.read_csv(csv_path)
        self.df = csv[csv['data_size'] == 32768]
        self.data_folder = images_folder
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        self.df.dropna()
        labels=self.df['transmitter_address'].iloc[index]

        #print(type(labels),labels.shape)
        #print(labels)
        label_map = {"Dev_1": 0, "Dev_2": 1}
        #print(label_map["Dev_1"])

        #labels=labels.map(label_map).to_numpy()

        #self.df['transmitter_address'].replace({"Dev_2": 2})

        #labels=list(map(lambda x:  label_map[labels] ,  labels))

        #idx = np.nonzero(label_map.keys() == labels[:, None])[1]
        #labels = np.asarray(label_map.values())[idx]

        #print("labels", labels)


        filename = self.df.iloc[index]["Burst_name"]
        mac_path= self.df.iloc[index]["transmitter_address"]
        antenna = self.df.iloc[index]["Antenna"]
        #print(type(labels))
        label=label_map[mac_path]
        #print(type(label), label)
        raw_signal = np.fromfile(os.path.join(self.data_folder, mac_path, filename + ".bin"), dtype="int16")
        size=int(len(raw_signal))

        print('len',size)

        #complex_sig= raw_signal[:size]+1j*raw_signal[size:]
        #signal_data=complex_sig[-512:]
        #real= torch.tensor(torch.from_numpy(raw_signal[:size]),dtype=torch.float64)
        #imag = torch.tensor(torch.from_numpy(raw_signal[size:]),dtype=torch.float64)
        #signal_complex=torch.complex64(real,imag)[:-512]
        signal_data=raw_signal[:]
        #print(type(signal_data), signal_data.shape)
        #signal_data = signal_data.astype(np.float)
        #signal_data=torch.from_numpy(signal_data)
        label = torch.tensor(label).float().unsqueeze(0)
        #label = torch.from_numpy(label)

        #print(type(signal_data), signal_data.shape)
        #print(type(label), label.shape)
        #IQ_signal = raw_signal[0::2] + 1j * raw_signal[1::2]
        #signal_data=IQ_signal[:1024]

        if self.transform :
            signal_data = self.transform(signal_data)
        if self.target_transform:
            label = self.target_transform(label)

        return signal_data, label

    def __convert_to_complex(self):
        pass
    def __load_samples(self):
        pass


class Lora_H5_dataset(torch.utils.data.Dataset):
    def __init__(self,data_path,transform = None,target_transform=None,convert_complex=True,data_start=None,data_end=None,is_train=False):
        self.file_path=data_path
        self.transform = transform
        self.target_transform = target_transform
        self.is_train = is_train
        self.label_name='label'
        self.data_name='data'
        self.h5file = h5py.File(self.file_path, 'r')
        label = self.h5file[self.label_name][data_start:data_end]
        self.data= self.h5file[self.data_name][data_start:data_end]
        if convert_complex:

            self.data= self._convert_to_complex(self.data)
        label = label.astype(int)

        #TODO

        label=np.transpose(label)
        print("before label change", np.unique(label), type(label), label[0])
        for ind in range(len(label)):
            if label[ind][0] >= 4:
                label[ind][0] = 1
            else:
                label[ind][0] = 0

        print("after label change", np.unique(label), type(label), label[0])


        self.label = label
        print("shapes", self.label.shape, self.data.shape)


        print("unique label : ", np.unique(label))


    def __len__(self):
        return len(self.label)


    def __getitem__(self, index):


        label= self.label[index]
        data = self.data[index]

        #print("label",label)

        #print("shapes",label.shape,data.shape)
        label = torch.tensor(label).float().unsqueeze(0)
        data = torch.tensor(data).float()

        #data= data.unsqueeze(0)

        #data=torch.from_numpy(data)
        #label = label.float().unsqueeze(0)

        #print("before transform",data[:10])
        #print(data[:],data.shape,type(data),type(data[1]))
        #data=np.reshape(data,(data.shape[0],1))
        #data = torch.from_numpy(data)
        #print(data[:], data.shape, type(data), type(data[1]))

        if self.is_train:
            anchor_label = self.label[index]

            positive_list = self.index[self.index != index][self.label[self.index != index] == anchor_label]

            positive_item = random.choice(positive_list)
            positive_img = self.data[positive_item]

            negative_list = self.index[self.index != index][self.label[self.index != index] != anchor_label]
            negative_item = random.choice(negative_list)
            negative_img = self.data[negative_item]

            if self.transform:
                anchor_img = self.transform(data)
                positive_img = self.transform(positive_img)
                negative_img = self.transform(negative_img)
            if self.target_transform is not None:
                label = self.target_transform(label)

            return anchor_img, positive_img, negative_img, anchor_label

        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            label = self.target_transform(label)
        #print("transformed",data[:10])
        #print("dsd")
        #print("texted",data.shape,label.shape)
        return data, label

    def _convert_to_complex(self,data):
        num_row = data.shape[0]
        num_col = data.shape[1]
        print(num_col)

        #data_complex = np.zeros([num_row, round(num_col / 2)], dtype=complex)

        data_complex = data[:, :round(num_col / 2)] + 1j * data[:, round(num_col / 2):]
        return data_complex