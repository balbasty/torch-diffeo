__all__ = ['Laplace', 'Helmoltz']
from diffeo.dft import FrequencyTransform
from diffeo.metrics.base import Metric
from . import greens


class Laplace(Metric):
    """
    Positive semi-definite metric based on the Laplace operator.
    This is relatively similar to SPM's "membrane" energy, but relies on
    the (ill-posed) analytical form of the Greens function.

    https://en.wikipedia.org/wiki/Laplace%27s_equation
    https://en.wikipedia.org/wiki/Green%27s_function
    """

    def __init__(self, factor=1, voxel_size=1, bound='dft',
                 learnable=False, cache=False):
        """
        Parameters
        ----------
        factor : float
            Regularization factor (optionally: learnable)
        voxel_size : list[float]
            Voxel size
        bound : {'dft', 'dct[1|2|3|4]', 'dst[1|2|3|4]'}
            Boundary conditions
        learnable : bool
            Make `factor` a learnable parameter
        cache : bool or int
            Cache up to `n` kernels
        """
        super().__init__(factor, voxel_size, bound, learnable, cache)
        self._metric_kernel = {}
        self._metric_fourier = {}
        self._greens_kernel = {}
        self._greens_fourier = {}

    def greens_kernel(self, x):
        shape = tuple(x.shape[2:])
        kernel = self._greens_kernel.get(shape, None)
        if kernel is None:
            kernel = greens.laplace(shape, self.voxel_size,
                                    dtype=x.dtype, device=x.device)
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


class Helmoltz(Metric):
    """
    Positive semi-definite metric based on the Helmoltz operator.
    This is relatively similar to SPM's mixture of "absolute" and
    "membrane" energies, but relies on the (ill-posed) analytical form
    of the Greens function.

    https://en.wikipedia.org/wiki/Helmholtz_equation
    https://en.wikipedia.org/wiki/Green%27s_function
    """

    def __init__(self, factor=1, alpha=1e-3, voxel_size=1, bound='dft',
                 learnable=False, cache=False):
        """
        Parameters
        ----------
        factor : float
            Regularization factor (optionally: learnable)
        alpha : float
            Diagonal regularizer.
            It is the square of the eigenvalue in the Helmoltz equation.
        voxel_size : list[float]
            Voxel size
        bound : {'dft', 'dct[1|2|3|4]', 'dst[1|2|3|4]'}
            Boundary conditions
        learnable : bool
            Make `factor` a learnable parameter
        cache : bool or int
            Cache up to `n` kernels
        """
        super().__init__(factor, voxel_size, bound, learnable, cache)
        self.alpha = alpha
        self._metric_kernel = {}
        self._metric_fourier = {}
        self._greens_kernel = {}
        self._greens_fourier = {}

    def greens_kernel(self, x):
        shape = tuple(x.shape[2:])
        kernel = self._greens_kernel.get(shape, None)
        if kernel is None:
            kernel = greens.helmoltz(shape, self.alpha, self.voxel_size,
                                     dtype=x.dtype, device=x.device)
            self.cachetensor(self._greens_kernel, shape, kernel)
        return kernel.to(x) / self.factor

    def greens_fourier(self, x):
        shape = tuple(x.shape[2:])
        kernel = self._greens_kernel.get(shape, None)
        if kernel is None:
            with self.no_cache():
                kernel = self.greens_kernel(x)
            ft = FrequencyTransform(len(shape), self.bound).forward_kernel
            kernel = ft(kernel.movedim(-1, 0)).movedim(0. -1)
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
            ft = FrequencyTransform(len(shape), self.bound).inverse_kernel
            kernel = ft(kernel.movedim(-1, 0)).movedim(0. -1)
            self.cachetensor(self._metric_kernel, shape, kernel)
        return kernel.to(x) / self.factor
