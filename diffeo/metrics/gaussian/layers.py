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

    def __init__(self, fwhm=16, factor=1, voxel_size=1, bound='circulant',
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
        bound : [list of] {'circulant', 'neumann', 'dirichlet', 'sliding'}
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
        learnable_factor = learnable_factor or (isinstance(learnable, bool) and learnable)
        super().__init__(factor, voxel_size, bound, learnable_factor, cache)
        if learnable_fwhm:
            self.fwhm = nn.Parameter(torch.as_tensor(fwhm), requires_grad=True)
        else:
            self.fwhm = fwhm
        if getattr(self.fwhm, 'requires_grad', False) and cache:
            raise ValueError('Cannot use both `cache=True` and a learnable fwhm')
        self._metric_kernel = {}
        self._metric_fourier = {}
        self._greens_kernel = {}
        self._greens_fourier = {}
        self.learnable = learnable

    def to_voxel_size(self, voxel_size):
        return type(self)(
            fwhm=self.fwhm,
            factor=self.factor,
            voxel_size=voxel_size,
            bound=self.bound,
            learnable=self.learnable,
            cache=self.cache,
        )

    def greens_kernel(self, x, factor=True):
        # x : (..., *spatial, D) tensor
        #  -> (*spatial, D, [D]) tensor
        ndim = x.shape[-1]
        shape = tuple(x.shape[-ndim-1:-1])
        kernel = self._greens_kernel.get(shape, None)
        if kernel is None:
            voxel_size = make_vector(self.voxel_size, dtype=x.dtype, device=x.device)
            kernel = torch.stack(cartesian_grid(shape, dtype=x.dtype, device=x.device), -1)
            kernel -= torch.as_tensor(shape, dtype=x.dtype, device=x.device).sub_(1).div_(2)
            kernel *= voxel_size
            kernel = kernel.square_().sum(-1)
            # now stop using inplace operations in case we need gradients
            lam = (2.355 / self.fwhm) ** 2
            kernel = (-lam * kernel).exp()
            kernel = kernel / kernel.sum()
            kernel = torch.stack([kernel]*len(shape), -1)
            kernel = kernel / voxel_size.square()
            self.cachetensor(self._greens_kernel, shape, kernel)
        kernel = kernel.to(x)
        if factor:
            kernel = kernel / self.factor
        return kernel

    def greens_fourier(self, x, factor=True):
        # x : (..., *spatial, D) tensor
        #  -> (*spatial, D, [D]) tensor
        ndim = x.shape[-1]
        shape = tuple(x.shape[-ndim-1:-1])
        kernel = self._greens_kernel.get(shape, None)
        if kernel is None:
            with self.no_cache():
                kernel = self.greens_kernel(x, factor=False)
            ft = FrequencyTransform(len(shape), self.bound, norm='ortho').forward_kernel
            kernel = ft(kernel)
            self.cachetensor(self._greens_fourier, shape, kernel)
        kernel = kernel.to(x)
        if factor:
            kernel = kernel / self.factor
        return kernel

    def metric_fourier(self, x, factor=True):
        # x : (..., *spatial, D) tensor
        #  -> (*spatial, D, [D]) tensor
        ndim = x.shape[-1]
        shape = tuple(x.shape[-ndim-1:-1])
        kernel = self._metric_fourier.get(shape, None)
        if kernel is None:
            with self.no_cache():
                kernel = self.greens_fourier(x, factor=False).reciprocal()
            self.cachetensor(self._metric_fourier, shape, kernel)
        kernel = kernel.to(x)
        if factor:
            kernel = kernel * self.factor
        return kernel

    def metric_kernel(self, x, factor=True):
        # x : (..., *spatial, D) tensor
        #  -> (*spatial, D, [D]) tensor
        ndim = x.shape[-1]
        shape = tuple(x.shape[-ndim-1:-1])
        kernel = self._metric_kernel.get(shape, None)
        if kernel is None:
            with self.no_cache():
                kernel = self.metric_fourier(x, factor=False)
            ft = FrequencyTransform(len(shape), self.bound, norm='ortho').inverse_kernel
            kernel = ft(kernel)
            self.cachetensor(self._metric_kernel, shape, kernel)
        kernel = kernel.to(x)
        if factor:
            kernel = kernel * self.factor
        return kernel

