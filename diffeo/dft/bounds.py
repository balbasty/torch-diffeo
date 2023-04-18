from torch import nn
import torch
from .generic import dtn, kerdtn, dtshift, real
from diffeo.bounds import bound2dft, has_sliding, sliding2dft
from diffeo.utils import ensure_list


msg_error_dct = """
DCT/DST require scipy (and, for cuda support, cupy)
Install torch-diffeo with `pip install "torch-diffeo[dct,cuda]"`
"""


class FrequencyTransform(nn.Module):
    """
    A multidimensional frequency transform that is compatible with
    common boundary conditions.
    """

    def __init__(self, ndim, bound='circulant'):
        """
        Parameters
        ----------
        ndim : int
            Number of spatial dimensions
        bound : {'circulant', 'neumann', 'dirichlet', 'sliding'}
            Boundary conditions.
        """
        super().__init__()
        self.ndim = ndim
        self.dims = list(range(-self.ndim, 0))

        # While the convolution theorem gives DFT * DFT -> DFT, this is
        # not true for the real variants DST/DCT, where specific
        # combinations of types must be used.
        # See https://en.wikipedia.org/wiki/Symmetric_convolution

        bound = ensure_list(bound, ndim)
        bound = list(map(lambda x: bound2dft.get(x, x), bound))
        self.bound = bound
        self.sym = [b != 'dft' for b in bound]
        self.sliding = has_sliding(bound)

    def forward(self, x):
        """x : (..., [D], *spatial) tensor"""
        if not self.sliding:
            return dtn(x, self.bound, dim=self.dims)
        else:
            x, y = torch.empty_like(x), x
            x = x.movedim(-self.ndim-1, 0)
            y = y.movedim(-self.ndim-1, 0)
            for i, d in enumerate(self.dims):
                x[i] = dtn(y[i], sliding2dft(self.bound, i), dim=self.dims)
            x = x.movedim(0, -self.ndim-1)
            return x

    def inverse(self, x):
        """x : (..., [D], *spatial) tensor"""
        if not self.sliding:
            return real(dtn(x, self.bound, dim=self.dims, inverse=True))
        else:
            # sliding
            x, y = torch.empty_like(x), x
            x = x.movedim(-self.ndim-1, 0)
            y = y.movedim(-self.ndim-1, 0)
            for i, d in enumerate(self.dims):
                x[i] = dtn(y[i], sliding2dft(self.bound, i), dim=self.dims,
                           inverse=True)
            x = x.movedim(0, -self.ndim-1)
            return x

    def forward_kernel(self, x):
        """x : (..., [D], D, *spatial) tensor"""
        x = dtshift(x, self.ndim, self.sym)
        x = kerdtn(x, self.ndim, self.sym)
        return x

    def inverse_kernel(self, x):
        """x : (..., [D], D, *spatial) tensor"""
        # FIXME
        x = dtshift(x, self.ndim, self.sym)
        x = kerdtn(x, self.ndim, self.sym, inverse=True)
        return x
