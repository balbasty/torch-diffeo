__all__ = [
    'dft', 'idft',
    'dct1', 'dct2', 'dct3', 'dct4', 'idct1', 'idct2', 'idct3', 'idct4',
    'dst1', 'dst2', 'dst3', 'dst4', 'idst1', 'idst2', 'idst3', 'idst4',
]
from torch import nn
import torch.fft
from math import floor
try:
    from .dct import *
    dct = dct2
    dst = dst2
except ImportError:
    dct = dct1 = dct2 = dct3 = dct4 = idct1 = idct2 = idct3 = idct4 = None
    dst = dst1 = dst2 = dst3 = dst4 = idst1 = idst2 = idst3 = idst4 = None

dft = torch.fft.fftn
idft = torch.fft.ifftn

msg_error_dct = """
DCT/DST require scipy (and, for cuda support, cupy)
Install torch-diffeo with `pip install "torch-diffeo[dct,cuda]"`
"""


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
            self._fwd_ker_off = dst1
            self._inv_ker_off = idst1
        elif bound == 'dct2':
            self._fwd = dct2
            self._inv = idct2
            self._fwd_ker = dct1
            self._inv_ker = idct1
            self._fwd_ker_off = dst1
            self._inv_ker_off = idst1
        elif bound == 'dct3':
            self._fwd = dct3
            self._inv = idct3
            self._fwd_ker = dct3
            self._inv_ker = idct3
            self._fwd_ker_off = dst3
            self._inv_ker_off = idst3
        elif bound == 'dct4':
            self._fwd = dct4
            self._inv = idct4
            self._fwd_ker = dct3
            self._inv_ker = idct3
            self._fwd_ker_off = dst3
            self._inv_ker_off = idst3
        elif bound == 'dst1':
            self._fwd = dst1
            self._inv = idst1
            self._fwd_ker = dct1
            self._inv_ker = idct1
            self._fwd_ker_off = dst1
            self._inv_ker_off = idst1
        elif bound == 'dst2':
            self._fwd = dst2
            self._inv = idst2
            self._fwd_ker = dct1
            self._inv_ker = idct1
            self._fwd_ker_off = dst1
            self._inv_ker_off = idst1
        elif bound == 'dst3':
            self._fwd = dst3
            self._inv = idst3
            self._fwd_ker = dct3
            self._inv_ker = idct3
            self._fwd_ker_off = dst3
            self._inv_ker_off = idst3
        elif bound == 'dst4':
            self._fwd = dst4
            self._inv = idst4
            self._fwd_ker = dct3
            self._inv_ker = idct3
            self._fwd_ker_off = dst3
            self._inv_ker_off = idst3
        else:
            raise ValueError(f'Unknown bound "{bound}"')
        if bound != 'dft' and dct is None:
            raise ValueError(msg_error_dct)
        self.bound = bound

    def forward(self, x):
        dims = list(range(-self.ndim, 0))
        x = self._fwd(x, dim=dims, norm=self.norm)
        return x

    def inverse(self, x):
        dims = list(range(-self.ndim, 0))
        x = real(self._inv(x, dim=dims, norm=self.norm))
        return x

    def forward_kernel_dft(self, x):
        dims = list(range(-self.ndim, 0))
        x = torch.fft.ifftshift(x, dims)
        dtype = x.dtype
        x = real(self._fwd_ker(x, dim=dims, norm=self.norm))
        return x.to(dtype)

    def inverse_kernel_dft(self, x):
        dims = list(range(-self.ndim, 0))
        x = torch.fft.ifftshift(x, dims)
        dtype = x.dtype
        x = real(self._inv_ker(x, dim=dims, norm=self.norm))
        return x.to(dtype)

    def forward_kernel(self, x):
        if self.bound == 'dft':
            return self.forward_kernel_dft(x)

        dims = list(range(-self.ndim, 0))
        x = sympart(x, self.ndim)
        x = torch.fft.ifftshift(x, dims)
        dtype = x.dtype
        x = kerft(x, self.ndim, self._fwd_ker, self._fwd_ker_off, self.norm)
        return x.to(dtype)

    def inverse_kernel(self, x):
        if self.bound == 'dft':
            return self.inverse_kernel_dft(x)

        dims = list(range(-self.ndim, 0))
        x = sympart(x, self.ndim)
        x = torch.fft.ifftshift(x, dims)
        dtype = x.dtype
        x = kerft(x, self.ndim, self._inv_ker, self._inv_ker_off, self.norm)
        return x.to(dtype)


def real(x):
    if x.is_complex():
        x = x.real
    return x


def sympart(x, ndim):

    # only keep the lower half part of the kernel
    # the rest can be reconstructed by symmetry and is
    # not needed by DCT

    # in the linear-elastic case, the off-diagonal elements
    # of the kernel have an odd-symmetry with zero center
    # and the symmetric part must therefore be shifted by
    # one voxel

    nbatch = x.ndim - ndim
    shape = x.shape[-ndim:]
    tmp, x = x, torch.zeros_like(x)
    slicer = tuple(slice(int(floor(s / 2)), None) for s in shape)
    if nbatch == 2:
        slicer0 = tuple(slice(int(floor(s / 2)), -1) for s in shape)
        slicer1 = tuple(slice(int(floor(s / 2)) + 1, None) for s in shape)
        for d in range(ndim):
            x[(d, d, *slicer)] = tmp[(d, d, *slicer)]
            for dd in range(ndim):
                if dd == d: continue
                x[(d, dd, *slicer0)] = tmp[(d, dd, *slicer1)]
    else:
        slicer = (slice(None),) * nbatch + slicer
        x[slicer] = tmp[slicer]
    return x


def kerft(x, ndim, diag, offdiag, norm):
    nbatch = x.ndim - ndim
    dims = list(range(-ndim, 0))
    if nbatch == 2:
        x, y = torch.zeros_like(x), x
        for d in range(ndim):
            x[d, d] = real(diag(y[d, d], dim=dims, norm=norm))
            for dd in range(ndim):
                if dd == d: continue
                x[d, dd] = real(offdiag(y[d, dd], dim=dims, norm=norm))
    else:
        x = real(diag(x, dim=dims, norm=norm))
    return x
