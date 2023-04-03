__all__ = [
    'dft', 'idft',
    'dct1', 'dct2', 'dct3', 'dct4', 'idct1', 'idct2', 'idct3', 'idct4',
    'dst1', 'dst2', 'dst3', 'dst4', 'idst1', 'idst2', 'idst3', 'idst4',
]
from torch import nn
import torch.fft
from .dct import *

dft = torch.fft.fftn
idft = torch.fft.ifftn


class FrequencyTransform(nn.Module):

    def __init__(self, ndim, bound='dft', norm=None):
        super().__init__()
        self.ndim = ndim
        self.norm = norm

        # While the convolution theorem gives DFT * DFT -> DFT, this is
        # not true for the real variants DST/DCT, where specific
        # combinations of types must be used.
        # See https://en.wikipedia.org/wiki/Symmetric_convolution

        bound = bound.lower()
        if bound == 'dft':
            self._fwd = dft
            self._inv = idft
            self._fwd_ker = dft
            self._inv_ker = idft
        elif bound == 'dct1':
            self._fwd = dct1
            self._inv = idct1
            self._fwd_ker = dct1
            self._inv_ker = idct1
        elif bound == 'dct2':
            self._fwd = dct2
            self._inv = idct2
            self._fwd_ker = dct1
            self._inv_ker = idct1
        elif bound == 'dct3':
            self._fwd = dct3
            self._inv = idct3
            self._fwd_ker = dct3
            self._inv_ker = idct3
        elif bound == 'dct4':
            self._fwd = dct4
            self._inv = idct4
            self._fwd_ker = dct3
            self._inv_ker = idct3
        elif bound == 'dst1':
            self._fwd = dst1
            self._inv = idst1
            self._fwd_ker = dct1
            self._inv_ker = idct1
        elif bound == 'dst2':
            self._fwd = dst2
            self._inv = idst2
            self._fwd_ker = dct1
            self._inv_ker = idct1
        elif bound == 'dst3':
            self._fwd = dst3
            self._inv = idst3
            self._fwd_ker = dct3
            self._inv_ker = idct3
        elif bound == 'dst4':
            self._fwd = dst4
            self._inv = idst4
            self._fwd_ker = dct3
            self._inv_ker = idct3
        else:
            raise ValueError(f'Unknown bound "{bound}"')

    def forward(self, x):
        dims = list(range(-self.ndim, 0))
        x = torch.fft.ifftshift(x, dims)
        x = self._fwd(x, dim=dims, norm=self.norm)
        x = torch.fft.fftshift(x, dims)
        return x

    def inverse(self, x):
        dims = list(range(-self.ndim, 0))
        x = torch.fft.ifftshift(x, dims)
        x = self._inv(x, dim=dims, norm=self.norm)
        if x.is_complex():
            x = x.real
        x = torch.fft.fftshift(x, dims)
        return x

    def forward_kernel(self, x):
        dims = list(range(-self.ndim, 0))
        x = torch.fft.ifftshift(x, dims)
        dtype = x.dtype
        x = self._fwd_ker(x, dim=dims, norm=self.norm)
        if x.is_complex():
            x = x.real
        x = torch.fft.fftshift(x.to(dtype), dims)
        return x

    def inverse_kernel(self, x):
        dims = list(range(-self.ndim, 0))
        x = torch.fft.ifftshift(x, dims)
        dtype = x.dtype
        x = self._inv_ker(x, dim=dims, norm=self.norm)
        if x.is_complex():
            x = x.real
        x = torch.fft.fftshift(x.to(dtype), dims)
        return x
