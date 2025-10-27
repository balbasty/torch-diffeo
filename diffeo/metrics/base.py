from contextlib import contextmanager
from torch import nn
import torch
from diffeo.dft import FrequencyTransform
import math


class Metric(nn.Module):

    def __init__(self, factor=1, voxel_size=1, bound='circulant',
                 learnable=False, cache=False):
        """
        Parameters
        ----------
        factor : float
            Regularization factor
        voxel_size : list[float]
            Voxel size
        bound : {'circulant', 'neumann', 'dirichlet', 'sliding'}
            Boundary conditions
        learnable : bool
            Make `factor` a learnable parameter
        cache : bool or int
            Cache up to `n` kernels
        """
        super().__init__()
        self.voxel_size = voxel_size
        self.bound = bound
        self.cache = cache
        if learnable:
            self.factor = nn.Parameter(torch.as_tensor(factor), requires_grad=True)
        else:
            self.factor = factor

    def cachetensor(self, cache, key, tensor):
        if self.cache:
            cache[key] = tensor
            if len(cache) > int(self.cache):
                del cache[cache.keys[0]]

    @contextmanager
    def no_cache(self):
        cache, self.cache = self.cache, False
        try:
            yield
        finally:
            self.cache = cache

    def forward(self, x, factor=True):
        """Apply the forward linear operator: v -> Lv
        v : (..., *spatial, D) tensor
        """
        ndim = x.shape[-1]
        ft = FrequencyTransform(ndim, self.bound)

        # Fourier kernel
        kernel = self.metric_fourier(x, factor=False)

        # Fourier transform
        x = ft.forward(x.movedim(-1, -ndim-1)).movedim(-ndim-1, -1)

        # Matrix multiply
        if kernel.ndim == ndim + 2:
            # matrix multiply
            if x.is_complex():
                x = torch.complex(
                    kernel.matmul(x.real.unsqueeze(-1)).squeeze(-1),
                    kernel.matmul(x.imag.unsqueeze(-1)).squeeze(-1))
            else:
                x = kernel.matmul(x.unsqueeze(-1)).squeeze(-1)
        else:
            # pointwise multiply
            x = x * kernel

        # Inverse Fourier transform
        x = real(ft.inverse(x.movedim(-1, -ndim-1)).movedim(-ndim-1, -1))

        # Global factor
        if factor:
            x = x * self.factor

        return x

    def inverse(self, x, factor=True):
        """Apply the inverse (Greens) linear operator: m -> Km
        m : (..., *spatial, D) tensor
        """
        ndim = x.shape[-1]
        ft = FrequencyTransform(ndim, self.bound)

        # Fourier kernel
        kernel = self.greens_fourier(x, factor=False)

        # Fourier transform
        x = ft.forward(x.movedim(-1, -ndim-1)).movedim(-ndim-1, -1)

        if kernel.ndim == ndim + 2:
            # matrix multiply
            if x.is_complex():
                x = torch.complex(
                    kernel.matmul(x.real.unsqueeze(-1)).squeeze(-1),
                    kernel.matmul(x.imag.unsqueeze(-1)).squeeze(-1))
            else:
                x = kernel.matmul(x.unsqueeze(-1)).squeeze(-1)
        else:
            # pointwise multiply
            x = x * kernel

        # Inverse Fourier transform
        x = real(ft.inverse(x.movedim(-1, -ndim-1)).movedim(-ndim-1, -1))

        # Global factor
        if factor:
            x = x / self.factor

        return x

    def whiten(self, x, factor=True):
        """Apply the inverse square root linear operator: v -> sqrt(K) v
        v : (..., *spatial, D) tensor
        """
        ndim = x.shape[-1]
        ft = FrequencyTransform(ndim, self.bound)

        # Fourier kernel
        kernel = self.greens_fourier(x, factor=False)

        # Square root
        if kernel.ndim == ndim + 2:
            # SVD
            eigval, eigvec = torch.linalg.eigh(kernel)
            eigval = eigval.sqrt()
            kernel = eigvec.matmul(eigval.unsqueeze(-1) * eigvec.transpose(-1, -2))
            del eigval, eigvec
        else:
            assert kernel.ndim in (x.ndim-1, x.ndim-2)
            # diagonal
            kernel = kernel.sqrt()

        # Fourier transform
        x = ft.forward(x.movedim(-1, -ndim-1)).movedim(-ndim-1, -1)

        if kernel.ndim == ndim + 2:
            # matrix multiply
            if x.is_complex():
                x = torch.complex(
                    kernel.matmul(x.real.unsqueeze(-1)).squeeze(-1),
                    kernel.matmul(x.imag.unsqueeze(-1)).squeeze(-1))
            else:
                x = kernel.matmul(x.unsqueeze(-1)).squeeze(-1)
        else:
            # pointwise multiply
            x = x * kernel

        # Inverse Fourier transform
        x = real(ft.inverse(x.movedim(-1, -ndim-1)).movedim(-ndim-1, -1))

        # Global factor
        if factor:
            x = x / (self.factor ** 0.5)

        return x

    def color(self, x, factor=True):
        """Apply the square root linear operator: m -> sqrt(L) m
        m : (..., *spatial, D) tensor
        """
        ndim = x.shape[-1]
        ft = FrequencyTransform(ndim, self.bound)

        # Fourier kernel
        kernel = self.metric_fourier(x, factor=False)

        # Square root
        if kernel.ndim == ndim + 2:
            # SVD
            eigval, eigvec = torch.linalg.eigh(kernel)
            eigvec = eigvec.sqrt()
            kernel = eigval.matmul(eigvec.unsqueeze(-1) * eigval.tranpose(-1, -2))
            del eigval, eigvec
        else:
            assert kernel.ndim in (x.ndim-1, x.ndim-2)
            # diagonal
            kernel = kernel.sqrt()

        # Fourier transform
        x = ft.forward(x.movedim(-1, -ndim-1)).movedim(-ndim-1, -1)

        if kernel.ndim == ndim + 2:
            # matrix multiply
            if x.is_complex():
                x = torch.complex(
                    kernel.matmul(x.real.unsqueeze(-1)).squeeze(-1),
                    kernel.matmul(x.imag.unsqueeze(-1)).squeeze(-1))
            else:
                x = kernel.matmul(x.unsqueeze(-1)).squeeze(-1)
        else:
            # pointwise multiply
            x = x * kernel

        # Inverse Fourier transform
        x = real(ft.inverse(x.movedim(-1, -ndim-1)).movedim(-ndim-1, -1))

        # Global factor
        if factor:
            x = x * (self.factor ** 0.5)

        return x

    def logdet(self, x, factor=True):
        """Return the log-determinant of L (times batch size)
        v : (..., *spatial, D) tensor
        """
        ndim = x.shape[-1]
        batch = x.shape[:-ndim-1]
        kernel = self.metric_fourier(x, factor=False)
        if kernel.ndim == ndim + 2:
            ld = kernel.logdet().sum() * batch.numel()
        else:
            ld = kernel.log().sum() * batch.numel()
        if factor:
            if torch.is_tensor(self.factor):
                logfactor = self.factor.log()
            else:
                logfactor = math.log(self.factor)
            ld += x.numel() * logfactor

    def greens_kernel(self, x, factor=True):
        raise NotImplementedError

    def greens_fourier(self, x, factor=True):
        raise NotImplementedError

    def metric_fourier(self, x, factor=True):
        raise NotImplementedError

    def metric_kernel(self, x, factor=True):
        raise NotImplementedError


def real(x):
    if x.is_complex():
        x = x.real
    return x
