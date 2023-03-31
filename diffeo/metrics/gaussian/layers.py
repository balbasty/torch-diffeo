__all__ = ['Gaussian']
import torch
from torch import nn
from diffeo.dft import FrequencyTransform
from diffeo.metrics.base import Metric
from diffeo.utils import cartesian_grid, make_vector


class Gaussian(Metric):
    """
    Positive semi-definite metric whose Greens function is a Gaussian filter.
    """

    def __init__(self, fwhm=16, factor=1, voxel_size=1, bound='dft',
                 learnable=False, cache=False):
        """
        Parameters
        ----------
        fwhm : float
            Full-width at half-maximum of the Gaussian filter, in mm
             (optionally: learnable)
        factor : float
            Global regularization factor (optionally: learnable)
        voxel_size : list[float]
            Voxel size
        bound : {'dft', 'dct[1|2|3|4]', 'dst[1|2|3|4]'}
            Boundary conditions
        learnable : bool or {'factor', 'fwhm', 'fwhm+factor}
            Make `factor` and/or 'fwhm' a learnable parameter.
            `True` is equivalent to `factor`.
        cache : bool or int
            Cache up to `n` kernels
            This cannot be used when `learnable='fwhm'`
        """
        learnable_fwhm = isinstance(learnable, str) and 'fwhm' in learnable
        learnable_factor = isinstance(learnable, str) and 'factor' in learnable
        learnable_factor = learnable_factor or isinstance(learnable, bool) and learnable
        if learnable_fwhm and cache:
            raise ValueError('Cannot use both `cache=True` and `learnable="fwhm"`')
        super().__init__(factor, voxel_size, bound, learnable_factor, cache)
        if learnable_fwhm:
            self.fwhm = nn.Parameter(torch.as_tensor(fwhm), requires_grad=True)
        else:
            self.fwhm = fwhm
        self._metric_kernel = {}
        self._metric_fourier = {}
        self._greens_kernel = {}
        self._greens_fourier = {}

    def greens_kernel(self, x):
        shape = tuple(x.shape[2:])
        kernel = self._greens_kernel.get(shape, None)
        if kernel is None:
            kernel = torch.stack(cartesian_grid(shape, dtype=x.dtype, device=x.device), -1)
            kernel -= torch.as_tensor(shape, dtype=x.dtype, device=x.device).sub_(1).div_(2)
            kernel *= make_vector(self.voxel_size, dtype=x.dtype, device=x.device)
            kernel = kernel.square_().sum(-1)
            # now stop using inplace operations in case we need gradients
            lam = (2.355 / self.fwhm) ** 2
            kernel = (-lam * kernel).exp()
            kernel = kernel / kernel.sum()
            self.cachetensor(self._greens_kernel, shape, kernel)
        return kernel.to(x) / self.factor

    def greens_fourier(self, x):
        shape = tuple(x.shape[2:])
        kernel = self._greens_kernel.get(shape, None)
        if kernel is None:
            with self.no_cache():
                kernel = self.greens_kernel(x)
            ft = FrequencyTransform(len(shape), self.bound, norm='ortho').forward_kernel
            kernel = ft(kernel)
            self.cachetensor(self._greens_fourier, shape, kernel)
        return kernel.to(x) / self.factor

    def metric_fourier(self, x):
        shape = tuple(x.shape[2:])
        kernel = self._metric_fourier.get(shape, None)
        if kernel is None:
            with self.no_cache():
                kernel = self.greens_fourier(x).reciprocal()
            self.cachetensor(self._metric_fourier, shape, kernel)
        return kernel.to(x) * self.factor

    def metric_kernel(self, x):
        shape = tuple(x.shape[2:])
        kernel = self._metric_kernel.get(shape, None)
        if kernel is None:
            with self.no_cache():
                kernel = self.metric_fourier(x)
            ft = FrequencyTransform(len(shape), self.bound, norm='ortho').inverse_kernel
            kernel = ft(kernel)
            self.cachetensor(self._metric_kernel, shape, kernel)
        return kernel.to(x) * self.factor

