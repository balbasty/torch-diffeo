from contextlib import contextmanager
from torch import nn
import torch
from diffeo.dft import FrequencyTransform


class Metric(nn.Module):

    def __init__(self, factor=1, voxel_size=1, bound='dft',
                 learnable=False, cache=False):
        """
        Parameters
        ----------
        factor : float
            Regularization factor
        voxel_size : list[float]
            Voxel size
        bound : {'dft', 'dct[1|2|3|4]', 'dst[1|2|3|4]'}
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

    def forward(self, x):
        """Apply the forward linear operator: v -> Lv"""
        ft = FrequencyTransform(x.ndim - 2, self.bound, norm='ortho')

        # Fourier kernel
        kernel = self.metric_fourier(x)

        # Fourier transform
        x = ft.forward(x)

        # Matrix multiply
        x = x * kernel

        # Inverse Fourier transform
        x = ft.inverse(x)

        return x

    def inverse(self, x):
        """Apply the inverse (Greens) linear operator: v -> Kv"""
        ft = FrequencyTransform(x.ndim - 2, self.bound, norm='ortho')

        # Fourier kernel
        kernel = self.greens_fourier(x)

        # Fourier transform
        x = ft.forward(x)

        # Matrix multiply
        x = x * kernel

        # Inverse Fourier transform
        x = ft.inverse(x)

        return x

    def whiten(self, x):
        """Apply the inverse square root linear operator: v -> sqrt(K) v"""
        ft = FrequencyTransform(x.ndim - 2, self.bound, norm='ortho')

        # Fourier kernel
        kernel = self.greens_fourier(x)

        # Square root
        if kernel.ndim == x.ndim:
            # SVD
            eigval, eigvec = torch.linalg.eigh(kernel)
            eigvec = eigvec.sqrt()
            kernel = eigval.matmul(eigvec.unqueeze(-1) * eigval.tranpose(-1, -2))
            del eigval, eigvec
        else:
            assert kernel.ndim in (x.ndim-1, x.ndim-2)
            # diagonal
            kernel = kernel.sqrt()

        # Fourier transform
        x = ft.forward(x)

        # Matrix multiply
        x = x * kernel

        # Inverse Fourier transform
        x = ft.inverse(x)

        return x

    def logdet(self, x):
        """Return the log-determinant of L (times batch size)"""
        nbatch = len(x)
        kernel = self.metric_fourier(x)
        if kernel.ndim == x.ndim:
            return kernel.logdet().sum() * nbatch
        else:
            return kernel.log().sum() * nbatch

    def greens_kernel(self, x):
        raise NotImplementedError

    def greens_fourier(self, x):
        raise NotImplementedError

    def metric_fourier(self, x):
        raise NotImplementedError

    def metric_kernel(self, x):
        raise NotImplementedError
