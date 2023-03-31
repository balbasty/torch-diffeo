__all__ = ['Mixture']
import torch
from torch import nn
from diffeo.dft import FrequencyTransform
from diffeo.metrics.base import Metric
from . import kernels, forward
from math import ceil, floor


class Mixture(Metric):
    """
    Positive semi-definite metric based on finite-difference regularisers.

    Mixture of "absolute", "membrane", "bending" and "linear-elastic" energies.
    Note that these quantities refer to what's penalised when computing the
    inner product (v, Lv). The "membrane" energy is therefore closely related
    to the "Laplacian" metric.
    """

    def __init__(self, absolute=0, membrane=0, bending=0, lame_shears=0, lame_div=0,
                 factor=1, voxel_size=1, bound='dft', use_diff=True,
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
        bound : {'dft', 'dct[1|2|3|4]', 'dst[1|2|3|4]'}
            Boundary conditions
        use_diff : bool
            Use finite differences to perform the forward pass.
            Otherwise, perform the convolution in Fourier space.
        learnable : bool or {'components'}
            Make `factor` a learnable parameter.
            If 'components', the individual factors (absolute, membrane, etc)
            are learned instead of the global factor, which is then fixed.
        cache : bool or int
            Cache up to `n` kernels
            This cannot be used when `learnable='components'`
        """
        learnable_components = isinstance(learnable, str) and learnable == 'components'
        if learnable_components:
            learnable = False
        if learnable_components and cache:
            raise ValueError('Cannot use both `cache=True` and `learnable="components"`')
        super().__init__(factor, voxel_size, bound, learnable, cache)
        if learnable_components:
            self.absolute = nn.Parameter(torch.as_tensor(absolute), requires_grad=True)
            self.membrane = nn.Parameter(torch.as_tensor(membrane), requires_grad=True)
            self.bending = nn.Parameter(torch.as_tensor(bending), requires_grad=True)
            self.lame_shears = nn.Parameter(torch.as_tensor(lame_shears), requires_grad=True)
            self.lame_div = nn.Parameter(torch.as_tensor(lame_div), requires_grad=True)
        else:
            self.absolute = absolute
            self.membrane = membrane
            self.bending = bending
            self.lame_shears = lame_shears
            self.lame_div = lame_div
        self._metric_kernel = {}
        self._metric_fourier = {}
        self._greens_kernel = {}
        self._greens_fourier = {}
        self.use_diff = use_diff

    def forward(self, x):
        if self.use_diff:
            return forward.mixture(
                x,
                self.absolute,
                self.membrane,
                self.bending,
                self.lame_shears,
                self.lame_div,
                self.factor,
                self.voxel_size,
                self.bound,
            )
        else:
            return super().forward(x)

    def is_linearelastic(self):
        return not is_zero(self.lame_shears) or not is_zero(self.lame_div)

    def metric_kernel(self, x):
        shape = tuple(x.shape[2:])
        kernel = self._metric_kernel.get(shape, None)
        if kernel is None:
            kernel = sum_kernel(self.absolute, self.membrane, self.bending,
                                self.leam_shears, self.lame_div, len(shape),
                                self.voxel_size, dtype=x.dtype, device=x.device)
            self.cachetensor(self._metric_kernel, shape, kernel)
        return kernel.to(x) * self.factor

    def metric_fourier(self, x):
        shape = tuple(x.shape[2:])
        kernel = self._metric_fourier.get(shape, None)
        if kernel is None:
            # Get compact kernel
            with self.no_cache():
                kernel0 = self.metric_kernel(x)

            # Embed kernel in full field of view
            kshape = [*shape, len(shape)]
            if self.is_linearelastic:
                kshape += [len(shape)]
            kernel = x.new_zeros(kshape)
            patch = tuple(slice(int(ceil((s-k)/2)), -int(floor((s-k)/2)) or None)
                          for s, k in zip(shape, kernel0.shape))
            kernel[patch].copy_(kernel0.to(kernel))

            # Fourier transform
            ft = FrequencyTransform(len(shape), self.bound, norm='ortho').forward_kernel
            kernel = kernel.movedim(-1, 0)
            if self.is_linearelastic():
                kernel = kernel.movedim(-1, 0)
            kernel = ft(kernel)
            kernel = kernel.movedim(0, -1)
            if self.is_linearelastic():
                kernel = kernel.movedim(0, -1)

            self.cachetensor(self._metric_fourier, shape, kernel)
        return kernel.to(x) * self.factor

    def greens_fourier(self, x):
        shape = tuple(x.shape[2:])
        kernel = self._greens_kernel.get(shape, None)
        if kernel is None:
            with self.no_cache():
                kernel = self.metric_fourier(x)
            kernel = kernel.reciprocal()
            self.cachetensor(self._greens_fourier, shape, kernel)
        return kernel.to(x) / self.factor

    def greens_kernel(self, x):
        shape = tuple(x.shape[2:])
        kernel = self._greens_kernel.get(shape, None)
        if kernel is None:
            with self.no_cache():
                kernel = self.greens_fourier(x)

            # Inverse Fourier transform
            ft = FrequencyTransform(len(shape), self.bound, norm='ortho').inverse_kernel
            kernel = kernel.movedim(-1, 0)
            if self.is_linearelastic():
                kernel = kernel.movedim(-1, 0)
            kernel = ft(kernel)
            kernel = kernel.movedim(0, -1)
            if self.is_linearelastic():
                kernel = kernel.movedim(0, -1)

            self.cachetensor(self._greens_kernel, shape, kernel)
        return kernel.to(x) / self.factor


def sum_kernel(a, m, b, s, d, ndim, vx, **backend):
    a0 = is_zero(a)
    m0 = is_zero(m)
    b0 = is_zero(b)
    s0 = is_zero(s)
    d0 = is_zero(d)

    k0 = 5 if b0 else 3 if (m0 or s0 or d0) else 1
    kshape = [k0] * ndim
    kshape += [ndim]
    if s0 or d0:
        kshape += [ndim]
    kernel = torch.zeros(kshape, **backend)

    if a0:
        patch = (slice((k0-1)//2, -(k0-1)//2 or None),) * ndim
        patch = kernel[patch]
        if s0 or d0:
            patch = patch.diagonal(0, -1, -2)
        patch.add_(kernels.absolute(ndim, vx, **backend).to_dense(), alpha=a)

    if m0:
        patch = (slice((k0-3)//2, -(k0-3)//2 or None),) * ndim
        patch = kernel[patch]
        if s0 or d0:
            patch = patch.diagonal(0, -1, -2)
        patch.add_(kernels.membrane(ndim, vx, **backend).to_dense(), alpha=m)

    if b0:
        patch = kernel
        if s0 or d0:
            patch = patch.diagonal(0, -1, -2)
        patch.add_(kernels.bending(ndim, vx, **backend).to_dense(), alpha=b)

    if s0:
        patch = (slice((k0-3)//2, -(k0-3)//2 or None),) * ndim
        patch = kernel[patch]
        patch.add_(kernels.lame_shear(ndim, vx, **backend).to_dense(), alpha=s)

    if d0:
        patch = (slice((k0-3)//2, -(k0-3)//2 or None),) * ndim
        patch = kernel[patch]
        patch.add_(kernels.lame_div(ndim, vx, **backend).to_dense(), alpha=d)

    return kernel


def is_zero(x):
    if torch.is_tensor(x) and x.requires_grad:
        return False
    else:
        return x == 0