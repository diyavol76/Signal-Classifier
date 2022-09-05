import h5py
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

h5_path=r"D:\iot\lora\Lora_Records\new_dataset\dataset\Diff_Days\dataset_220620.h5"

file=h5py.File(h5_path, "r")
print(file.values())
print(file.items())

print(file['data'][0])
raw=file['data'][0]
size=int(raw.size/2)


def normalization(data):
    ''' Normalize the signal.'''
    s_norm = np.zeros(data.shape)

    for i in range(data.shape[0]):
        sig_amplitude = np.abs(data[i])
        rms = np.sqrt(np.mean(sig_amplitude ** 2))
        s_norm[i] = data[i] / rms

    return s_norm

for i in range(20):
    raw = file['data'][i]
    complex=raw[:size]+1j*raw[size:]
    print(raw[:size])
    print(raw[size:])
    print(complex)
    complex=normalization(complex)

    ff,tt,sxx=signal.spectrogram(complex,fs=1000000,nfft=256,noverlap=128,nperseg=256)
    shifted= np.fft.fftshift(sxx, axes=0)
    #chan_ind_spec_amp = np.log10(np.abs(shifted) ** 2)


    f, t, spec = signal.stft(complex,
                             window='boxcar',
                             nperseg=256,
                             noverlap=128,
                             nfft=256,
                             return_onesided=False,
                             padded=False,
                             boundary=None)

    # FFT shift to adjust the central frequency.
    spec = np.fft.fftshift(spec, axes=0)

    # Generate channel independent spectrogram.
    chan_ind_spec = spec[:, 1:] / spec[:, :-1]

    # Take the logarithm of the magnitude.
    chan_ind_spec_amp = np.log10(np.abs(chan_ind_spec) ** 2)
    #plt.pcolormesh(t, f, chan_ind_spec_amp, shading='gouraud')
    data_channel_ind_spec = np.zeros([ 256, 62, 1])
    data_channel_ind_spec[ :, :, 0] = chan_ind_spec_amp
    #chan_ind_spec_amp[0]
    print(chan_ind_spec_amp[0])
    print(chan_ind_spec_amp.shape)
    plt.plot(chan_ind_spec_amp)

    plt.show()
