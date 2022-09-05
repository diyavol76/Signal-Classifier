import torch
from typing import Callable, Optional
from torch import Tensor
from scipy import signal
import numpy as np


class Wavelet_RF(torch.nn.Module):

    def __init__(self,
                 n_fft: int = 400,
                 win_length: Optional[int] = None,
                 hop_length: Optional[int] = None,
                 pad: int = 0,
                 window_fn: Callable[..., Tensor] = torch.hann_window,
                 power: Optional[float] = 2.,
                 normalized: bool = False,
                 wkwargs: Optional[dict] = None) -> None:
        super(Wavelet_RF, self).__init__()


    def forward(self):
        pass


class STFT_RF(torch.nn.Module):

    def __init__(self,
                 n_fft: int = 400,
                 win_length: Optional[int] = None,
                 hop_length: Optional[int] = None,
                 pad: int = 0,
                 window_fn: Callable[..., Tensor] = torch.hann_window,
                 power: Optional[float] = 2.,
                 normalized: bool = False,
                 wkwargs: Optional[dict] = None) -> None:
        super(Wavelet_RF, self).__init__()
        self.n_fft=n_fft
        # number of FFT bins. the returned STFT result will have n_fft // 2 + 1
        # number of frequecies due to onesided=True in torch.stft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        window = window_fn(self.win_length) if wkwargs is None else window_fn(self.win_length, **wkwargs)
        self.register_buffer('window', window)
        self.pad = pad
        self.power = power
        self.normalized = normalized

    def forward(self, waveform: Tensor) -> Tensor:
        # Short-time Fourier transform (STFT).
        f, t, spec = signal.stft(waveform,
                                 window='boxcar',
                                 nperseg=self.win_length,
                                 noverlap=self.hop_length,
                                 nfft=self.n_fft,
                                 return_onesided=False,
                                 padded=False,
                                 boundary=None)

        # FFT shift to adjust the central frequency.
        spec = np.fft.fftshift(spec, axes=0)

        # Generate channel independent spectrogram.
        chan_ind_spec = spec[:, 1:] / spec[:, :-1]

        # Take the logarithm of the magnitude.
        chan_ind_spec_amp = np.log10(np.abs(chan_ind_spec) ** 2)

        return chan_ind_spec_amp


class convert_to_complex_H5(torch.nn.Module):

    def __init__(self):
        super(convert_to_complex_H5, self).__init__()


    def forward(self,data):
        self.I_Q_seq = data
        #num_row = self.I_Q_seq.shape[0]
        num_col = self.I_Q_seq.shape[0]
        #print("num_col",num_col)

        #data_complex = np.zeros([num_row, round(num_col / 2)], dtype=complex)

        data_complex = self.I_Q_seq[ :round(num_col / 2)] + 1j * self.I_Q_seq[ round(num_col / 2):]
        return data_complex