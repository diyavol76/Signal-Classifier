import Lora.Lora_Dataset as Lora_Dataset
import pandas as pd
from torch.utils.data import DataLoader
import torchaudio
import matplotlib.pyplot  as plt
import numpy as np
#import torchvision.transforms.functional as F
import torchaudio.functional as F
from scipy import signal
import librosa
import librosa.display
import torch

csv_path=r"D:\iot\lora\Lora_Records\new_dataset\concatenated\concatenated.csv"
csv_path=r"D:\iot\lora\Lora_Records\new_dataset\concatenated\with_size_out.csv"
file_path=r"D:\iot\lora\Lora_Records\new_dataset\concatenated"

lora_h5_data_path= r"D:\iot\lora\Lora_Records\new_dataset\dataset\Diff_Days\dataset_220620.h5"
lora_h5_data_path= r"D:\iot\lora\Lora_Records\new_dataset\dataset_220627.h5"
lora_h5_data_path=r"D:\iot\data\net\lora-rffi\dataset\Test\dataset_residential.h5"
#lora_h5_data_path=r'D:\iot\lora\new_dataset\dataset\Diff_Days\dataset_220816.h5'
#lora_h5_data_path=r'D:\iot\data\net\lora-rffi\dataset\Train\dataset_training_aug.h5'
SAMPLE_RATE=1000000

def get_periodogram_psd_with_len(sig,sig_len):

    from_fft = np.fft.fft(sig,n=sig_len)

    shifted = np.fft.fftshift(np.abs(from_fft))

    output_signal = 20 * np.log10(shifted)

    return output_signal

#csv=pd.read_csv(csv_path)
#csv=csv[csv['data_size'] == 32768]
if __name__ == '__main__':

    transformation=torchaudio.transforms.Spectrogram()
    sector_size=256

    mel_spectrogram_lora = torchaudio.transforms.Spectrogram(
            #sample_rate=SAMPLE_RATE,
            n_fft=sector_size,
            hop_length=sector_size,
            #n_mels=sector,
            win_length=sector_size,
            window_fn=torch.hamming_window
        )


    #to_DB = torchaudio.transforms.AmplitudeToDB()

    #train_dataset=Lora_Dataset.RF_Dataset(csv_path,file_path,transform=None)
    train_dataset=Lora_Dataset.Lora_H5_dataset(lora_h5_data_path,transform=None,convert_complex=False)

    #image, label = train_dataset[0]
    #print(label)
    #print(train_dataset[10])

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    iter_data=iter(train_dataloader)
    #iter_data = torch.from_numpy(iter_data)
    while True:
        train_features, train_labels = next(iter_data)
        print(f"Feature batch shape: {train_features.size()}")
        print(f"Labels batch shape: {train_labels.size()}")
        img = train_features[0]
        label = train_labels[0]

        print(f"Label: {img}")

        print(f"Label: {label}")
        print(type(img))

        #num_signal=img.to('cpu').numpy()
        num_signal=img
        print("tensor_shape " ,num_signal.shape,label.shape)
        total_size=int(num_signal.shape[0]/2)
        #print("new size", total_size)
        #print(num_signal)

        #print(num_signal[:total_size])
        #print(num_signal[total_size:])

        #nfft
        nfft = 256

        #num_signal = torch.tensor(num_signal).float().unsqueeze(0)
        print(type(img),img[0])
        melll1 = torchaudio.transforms.Spectrogram(n_fft=128)(img)
        melll2 = torch.transpose(melll1, 0, 1)
        melll = np.fft.fftshift(melll2, axes=0)
        #tensor_spec, tensor_f, tensor_t = F.spectrogram(img, n_fft=nfft, window=torch.hann_window, pad=0,
         #                                               hop_length=int(nfft / 2), win_length=nfft, power=2,
        #                                                normalized=False)
        #melll = torch.fft.fftshift(melll)

        num_signal = img.to('cpu').numpy()



        print(type(num_signal), num_signal[0])
        I = num_signal[:total_size]
        Q = num_signal[total_size:]
        num_signal = num_signal[:total_size] + 1j * num_signal[total_size:]
        print("num", num_signal[3000:3050])

        #t_cwt = np.linspace(0, 1, 200)
        widths = np.arange(1, 4)
        cwtmatr = signal.cwt(num_signal, signal.ricker, widths)
        #num_signal = num_signal[:total_size]
        #num_signal = num_signal[0::2] + 1j * num_signal[1::2]
        #num_signal=np.abs(num_signal)

        #num_signal=num_signal[:total_size]
        #print("new size", num_signal.size())
        #print(num_signal)
        sector_size=128
#        torch_complex_signal=torch.from_numpy(num_signal)
        f, t, spec=signal.stft(num_signal,
                            window='boxcar',
                            nperseg= 128,
                            noverlap= 0,
                            nfft= 128,
                               return_onesided=False)

        spec = np.fft.fftshift(spec, axes=0)

        ff,tt,sxx=signal.spectrogram(num_signal,fs=1000000,nfft=nfft,noverlap=0,nperseg=128,mode='magnitude')


        #chan_ind_spec = spec[:, 1:] / spec[:, :-1]
        #sxx = np.fft.fftshift(sxx, axes=0)
        #print("ff",ff)
        #ff=ff[int(nfft/4):int(3*nfft/4)]
        #sxx=sxx[int(nfft/4):int(3*nfft/4)]
        print("spec shape ",np.array(sxx).shape)


        psd=get_periodogram_psd_with_len(num_signal,16384)
        #plt.plot(psd)
        #shifted= np.fft.fftshift(sxx, axes=0)
        #psd_shifted=get_periodogram_psd_with_len(np.fft.fftshift(spec, axes=0),1024)


        fig, axs = plt.subplots(5, figsize=(12, 8))
        new_spec=spec[50:75][:]

        #mel_specto,f3,t3=mel_spectrogram_lora(num_signal)
        #melll=torchaudio.transforms.Spectrogram(n_fft=128)(num_signal)

        axs[0].plot(psd)
        axs[1].pcolormesh(tt,ff,sxx,shading='gouraud')
        #axs[2].pcolormesh(t,f,np.abs(spec),shading='gouraud')
        axs[2].plot(np.transpose(np.abs(new_spec)))
        #axs[3].plot(m elll)
        #for i in range(10):
        #    axs[3].plot(spec[32+i][:])
        #print(new_spec)
        print("cwtshape", cwtmatr[:][:].shape)
        #axs[4].plot(np.transpose(cwtmatr))
        axs[3].plot(I)
        #axs[3].plot(Q)
        #axs[4].plot(I)
        #ff,tt,sxx=signal.spectrogram(num_signal,fs=1000000,nfft=256,noverlap=64,nperseg=128,mode='complex')
        axs[4].specgram(num_signal,Fs=1000000,noverlap=64,NFFT=256)
        #axs[4].pcolormesh(tt,ff,sxx,shading='flat')

        print(spec[:][:].shape)
        print(new_spec.shape)
        #print(new_spec)
        print(spec[10].shape)

        plt.show()