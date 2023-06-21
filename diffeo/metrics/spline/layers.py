__all__ = ['SplineMixture']
import torch
from torch import nn, Tensor
from math import floor
from typing import Union
from diffeo.dft import FrequencyTransform
from diffeo.metrics.base import Metric
from diffeo.linalg import batchinv
from diffeo.conv import flowconv
from . import kernels, forward


class SplineMixture(Metric):
    """
    Positive semi-definite metric based on analytical spline regularisers.

    Mixture of "absolute", "membrane", "bending" and "linear-elastic" energies.
    Note that these quantities refer to what's penalised when computing the
    inner product (v, Lv). The "membrane" energy is therefore closely related
    to the "Laplacian" metric.
    """

    def __init__(self, absolute=0, membrane=0, bending=0, lame_shears=0, lame_div=0,
                 factor=1, voxel_size=1, bound='circulant', order=3, use_conv=True,
                 learnable=False, cache=False):
        """
        Parameters
        ----------
        absolute : float
            Penalty on (squared) absolute values
        membrane: float
            Penalty on (squared) first derivatives
        bending : float
            Penalty on (squared) second derivatives
        lame_shears : float
            Penalty on the (squared) symmetric component of the Jacobian
        lame_div : float
            Penalty on the trace of the Jacobian
        factor : float
            Global regularization factor (optionally: learnable)
        voxel_size : list[float]
            Voxel size
        bound : [list of] {'circulant', 'neumann', 'dirichlet', 'sliding'}
            Boundary conditions
        order : int
            Spline order
        use_conv : bool
            Use convolution with small kernel to perform the forward pass.
            Otherwise, perform the convolution in Fourier space.
        learnable : bool or {'factor', 'components', 'factor+components'}
            Make `factor` a learnable parameter.
            If 'components', the individual factors (absolute, membrane, etc)
            are learned instead of the global factor.
            `True` is equivalent to `factor`.
        cache : bool or int
            Cache up to `n` kernels
            This cannot be used when `learnable='components'`
        """
        learnable_components = isinstance(learnable, str) and 'components' in learnable
        learnable_factor = isinstance(learnable, str) and 'factor' in learnable
        learnable_factor = learnable_factor or (isinstance(learnable, bool) and learnable)
        if not learnable_components:
            learnable_absolute = isinstance(learnable, str) and 'absolute' in learnable
            learnable_membrane = isinstance(learnable, str) and 'membrane' in learnable
            learnable_bending = isinstance(learnable, str) and 'bending' in learnable
            learnable_shears = isinstance(learnable, str) and 'shears' in learnable
            learnable_div = isinstance(learnable, str) and 'div' in learnable
        else:
            learnable_absolute = True
            learnable_membrane = True
            learnable_bending = True
            learnable_shears = True
            learnable_div = True
        super().__init__(factor, voxel_size, bound, learnable_factor, cache)
        if learnable_absolute:
            self.absolute = torch.as_tensor(absolute)
            if not self.absolute.dtype.is_floating_point:
                self.absolute = self.absolute.to(torch.get_default_dtype())
            self.absolute = nn.Parameter(self.absolute, requires_grad=True)
        else:
            self.absolute = absolute
        if learnable_membrane:
            self.membrane = torch.as_tensor(membrane)
            if not self.membrane.dtype.is_floating_point:
                self.membrane = self.membrane.to(torch.get_default_dtype())
            self.membrane = nn.Parameter(self.membrane, requires_grad=True)
        else:
            self.membrane = membrane
        if learnable_bending:
            self.bending = torch.as_tensor(bending)
            if not self.bending.dtype.is_floating_point:
                self.bending = self.bending.to(torch.get_default_dtype())
            self.bending = nn.Parameter(self.bending, requires_grad=True)
        else:
            self.bending = bending
        if learnable_shears:
            self.lame_shears = torch.as_tensor(lame_shears)
            if not self.lame_shears.dtype.is_floating_point:
                self.lame_shears = self.lame_shears.to(torch.get_default_dtype())
            self.lame_shears = nn.Parameter(self.lame_shears, requires_grad=True)
        else:
            self.lame_shears = lame_shears
        if learnable_div:
            self.lame_div = torch.as_tensor(lame_div)
            if not self.lame_div.dtype.is_floating_point:
                self.lame_div = self.lame_div.to(torch.get_default_dtype())
            self.lame_div = nn.Parameter(self.lame_div, requires_grad=True)
        else:
            self.lame_div = lame_div
        if cache and (
                getattr(self.absolute, 'requires_grad', False) or
                getattr(self.membrane, 'requires_grad', False) or
                getattr(self.bending, 'requires_grad', False) or
                getattr(self.lame_shears, 'requires_grad', False) or
                getattr(self.lame_div, 'requires_grad', False)
        ):
            raise ValueError('Cannot use both `cache=True` and learnable components')
        self._metric_kernel = {}
        self._metric_fourier = {}
        self._greens_kernel = {}
        self._greens_fourier = {}
        self.learnable = learnable
        self.use_conv = use_conv
        self.order = order

    def to_voxel_size(self, voxel_size):
        return type(self)(
            absolute=self.absolute,
            membrane=self.membrane,
            bending=self.bending,
            lame_shears=self.lame_shears,
            lame_div=self.lame_div,
            factor=self.factor,
            voxel_size=voxel_size,
            bound=self.bound,
            order=self.order,
            use_conv=self.use_conv,
            learnable=self.learnable,
            cache=self.cache,
        )

    def _str_opt(self):
        def make_opt(key):
            value = getattr(self, key)
            if torch.is_tensor(value):
                if value.numel() == 1:
                    value = value.item()
                else:
                    value = value.tolist()
            if isinstance(getattr(self, key), nn.Parameter):
                value = f'Parameter({value})'
            if value:
                return f'{key}={value}'
            else:
                return ''
        keys = ['factor', 'absolute', 'membrane', 'bending',
                'lame_shears', 'lame_div', 'bound']
        opt = ', '.join(map(make_opt, filter(bool, keys)))
        return opt

    def forward(self, x, factor=True):
        # x: (..., *spatial, D) tensor
        # -> (..., *spatial, D) tensor
        if self.use_conv:
            return forward.mixture(
                x,
                absolute=self.absolute,
                membrane=self.membrane,
                bending=self.bending,
                lame_shear=self.lame_shears,
                lame_div=self.lame_div,
                factor=self.factor if factor else 1,
                voxel_size=self.voxel_size,
                bound=self.bound,
                order=self.order,
            )
        else:
            return super().forward(x, factor)

    @property
    def is_linearelastic(self):
        return not is_zero(self.lame_shears) or not is_zero(self.lame_div)

    def metric_kernel(self, x, factor=True):
        # x : (..., *spatial, D) tensor
        #  -> (*spatial, D, [D]) tensor
        ndim = x.shape[-1]
        shape = tuple(x.shape[-ndim-1:-1])
        kernel = self._metric_kernel.get(shape, None)
        if kernel is None:
            kernel = kernels.mixture(
                ndim, self.absolute, self.membrane, self.bending,
                self.lame_shears, self.lame_div, 1, self.voxel_size,
                self.order, dtype=x.dtype, device=x.device,
            )
            self.cachetensor(self._metric_kernel, shape, kernel)
        kernel = kernel.to(x)
        if factor:
            kernel = kernel * self.factor
        return kernel

    def metric_fourier(self, x, factor=True):
        # x : (..., *spatial, D) tensor
        #  -> (*spatial, D, [D]) tensor
        ndim = x.shape[-1]
        shape = tuple(x.shape[-ndim-1:-1])
        kernel = self._metric_fourier.get(shape, None)
        if kernel is None:
            # Get compact kernel
            with self.no_cache():
                kernel0 = self.metric_kernel(x, factor=False)

            # Embed kernel in full field of view
            kshape = [*shape, len(shape)]
            if self.is_linearelastic:
                kshape += [len(shape)]
            kernel = x.new_zeros(kshape)
            patch = tuple(slice(int(floor(s/2))-k//2, int(floor(s/2))+k//2+1)
                          for s, k in zip(shape, kernel0.shape))
            kernel[patch].copy_(kernel0.to(kernel))

            # Fourier transform
            ft = FrequencyTransform(ndim, self.bound).forward_kernel
            kernel = kernel.movedim(-1, 0)
            if self.is_linearelastic:
                kernel = kernel.movedim(-1, 0)
            kernel = ft(kernel)
            kernel = kernel.movedim(0, -1)
            if self.is_linearelastic:
                kernel = kernel.movedim(0, -1)

            self.cachetensor(self._metric_fourier, shape, kernel)
        kernel = kernel.to(x)
        if factor:
            kernel = kernel * self.factor
        return kernel

    def greens_fourier(self, x, factor=True):
        # x : (..., *spatial, D) tensor
        #  -> (*spatial, D, [D]) tensor
        ndim = x.shape[-1]
        shape = tuple(x.shape[-ndim-1:-1])
        kernel = self._greens_kernel.get(shape, None)
        if kernel is None:
            with self.no_cache():
                kernel = self.metric_fourier(x, factor=False)
            if self.is_linearelastic:
                kernel = batchinv(kernel)
            else:
                kernel = kernel.reciprocal()
            self.cachetensor(self._greens_fourier, shape, kernel)
        kernel = kernel.to(x)
        if factor:
            kernel = kernel / self.factor
        return kernel

    def greens_kernel(self, x, factor=True):
        # x : (..., *spatial, D) tensor
        #  -> (*spatial, D, [D]) tensor
        ndim = x.shape[-1]
        shape = tuple(x.shape[-ndim-1:-1])
        kernel = self._greens_kernel.get(shape, None)
        if kernel is None:
            with self.no_cache():
                kernel = self.greens_fourier(x, factor=False)

            # Inverse Fourier transform
            ft = FrequencyTransform(ndim, self.bound).inverse_kernel
            kernel = kernel.movedim(-1, 0)
            if self.is_linearelastic:
                kernel = kernel.movedim(-1, 0)
            kernel = ft(kernel)
            kernel = kernel.movedim(0, -1)
            if self.is_linearelastic:
                kernel = kernel.movedim(0, -1)

            self.cachetensor(self._greens_kernel, shape, kernel)
        kernel = kernel.to(x)
        if factor:
            kernel = kernel / self.factor
        return kernel


def is_zero(x: Union[int, float, Tensor]) -> bool:
    if torch.is_tensor(x) and x.requires_grad:
        return False
    else:
        return x == 0
