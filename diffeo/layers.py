__all__ = [
    'Exp', 'ExpInv', 'ExpBoth',
    'Shoot', 'ShootInv', 'ShootBoth',
    'Compose', 'BCH',
    'Pull', 'Push', 'Count', 'ToCoeff', 'FromCoeff',
    'Upsample', 'Downsample', 'UpsampleFlow', 'DownsampleFlow',
]
from torch import nn
from diffeo.svf import exp, bch
from diffeo.shoot import shoot, default_metric
from diffeo.backends import default_backend
from diffeo.flows import compose
from diffeo.resize import upsample, downsample, upsample_flow, downsample_flow


class Exp(nn.Module):
    """
    Exponentiate a Stationary Velocity Field

    Returns the forward transform.
    """

    def __init__(self, bound='circulant', order=1, steps=8, anagrad=False, backend=None):
        """
        Parameters
        ----------
        bound : [list of] {'circulant', 'neumann', 'dirichlet', 'sliding'}
            Boundary conditions.
        order : int
            Order of encoding splines.
        steps : int
            Number of scaling and squaring steps.
        anagrad : bool
            Use analytical gradients instead of autograd.
        backend : module
            Backend to use to implement resampling.
            Must be one of the modules under `diffeo.backends`.

        Notes
        -----
        The number of equivalent Euler integration steps is `2**steps`

        Analytical gradients use less memory than autograd gradients,
        as they do not require storing intermediate time steps during
        scaling and squaring. However, they may be slightly less accurate.

        In differential equation terms, autograd corresponds to the
        strategy "discretize then optimize", whereas analytical gradients
        correspond to the strategy "optimize then discretize".
        """
        super().__init__()
        self.bound = bound
        self.order = order
        self.steps = steps
        self.anagrad = anagrad
        self.backend = backend or default_backend

    def forward(self, v):
        """
        Parameters
        ----------
        v : (B, D, *shape) tensor
            Stationary velocity field (in voxels)

        Returns
        -------
        phi : (B, D, *shape) tensor
            Displacement field (in voxels)
        """
        v = v.movedim(1, -1)
        v = exp(
            v,
            steps=self.steps, bound=self.bound, order=self.order,
            anagrad=self.anagrad, backend=self.backend,
        )
        v = v.movedim(-1, 1)
        return v


class ExpInv(Exp):
    """
    Exponentiate a Stationary Velocity Field

    Returns the inverse transform.
    """

    def forward(self, v):
        """
        Parameters
        ----------
        v : (B, D, *shape) tensor
            Stationary velocity field (in voxels)

        Returns
        -------
        phi : (B, D, *shape) tensor
            Inverse displacement field (in voxels)
        """
        return super().forward(-v)


class ExpBoth(Exp):
    """
    Exponentiate a Stationary Velocity Field

    Returns the forward and inverse transforms.
    """

    def forward(self, v):
        """
        Parameters
        ----------
        v : (B, D, *shape) tensor
            Stationary velocity field (in voxels)

        Returns
        -------
        phi : (B, D, *shape) tensor
            Inverse displacement field (in voxels)
        """
        return super().forward(v), super().forward(-v)


class BCH(nn.Module):
    """
    Compose two Stationary Velocity Fields using the BCH formula

    The Baker–Campbell–Hausdorff (BCH) allows computing z such that
    exp(z) = exp(x) o exp(y).

    https://en.wikipedia.org/wiki/BCH_formula
    """

    def __init__(self, bound='circulant', trunc=2, order=1, backend=None):
        """
        Parameters
        ----------
        bound : [list of] {'circulant', 'neumann', 'dirichlet', 'sliding'}
            Boundary conditions.
        trunc : int
            Maximum order used in the BCH series
        order : int
            Order of encoding splines.
        backend : module
            Backend to use to implement resampling.
            Must be one of the modules under `diffeo.backends`.
        """
        super().__init__()
        self.bound = bound
        self.trunc = trunc
        self.order = order
        self.backend = backend or default_backend

    def forward(self, left, right):
        """
        Parameters
        ----------
        left : (B, D, *shape) tensor
            Stationary velocity field (in voxels)
        right : (B, D, *shape) tensor
            Stationary velocity field (in voxels)

        Returns
        -------
        vel : (B, D, *shape) tensor
            Composed SVF (in voxels)

        """
        left = left.movedim(1, -1)
        right = right.movedim(1, -1)
        v = bch(
            left, right,
            trunc=self.trunc, bound=self.bound,
            order=self.order, backend=self.backend,
        )
        v = v.movedim(-1, 1)
        return v


class ShootBoth(nn.Module):
    """
    Exponentiate an Initial Velocity using Geodesic Shooting

    Returns the forward and inverse transform.
    """

    def __init__(self, metric=default_metric, steps=None, fast=True,
                 backend=None):
        """
        Parameters
        ----------
        metric : Metric
            A Riemannian metric
        steps : int
            Number of Euler integration steps.
            If None, use an educated guess based on the magnitude of
            the initial velocity.
        fast : int
            Use a faster but slightly less accurate integration scheme.
        backend : module
            Backend to use to implement pullback and pushforward.
            Must be one of the modules under `diffeo.backends`.
        """
        super().__init__()
        self.metric = metric
        self.steps = steps
        self.fast = fast
        self.backend = backend or default_backend

    def forward(self, v):
        """
        Parameters
        ----------
        v : (B, D, *shape) tensor
            Initial velocity field (in voxels)

        Returns
        -------
        phi : (B, D, *shape) tensor
            Forward displacement field (in voxels)
        iphi : (B, D, *shape) tensor
            Inverse displacement field (in voxels)
        """
        phi, iphi = shoot(
            v.movedim(1, -1),
            metric=self.metric, steps=self.steps,
            fast=self.fast, backend=self.backend,
        )
        return phi.movedim(-1, 1), iphi.movedim(-1, 1)


class Shoot(ShootBoth):
    """
    Exponentiate an Initial Velocity using Geodesic Shooting

    Returns the forward transform.
    """

    def forward(self, v):
        """
        Parameters
        ----------
        v : (B, D, *shape) tensor
            Initial velocity field (in voxels)

        Returns
        -------
        phi : (B, D, *shape) tensor
            Displacement field (in voxels)
        """
        return super().forward(v)[0]


class ShootInv(ShootBoth):
    """
    Exponentiate an Initial Velocity using Geodesic Shooting

    Returns the inverse transform.
    """

    def forward(self, v):
        """
        Parameters
        ----------
        v : (B, D, *shape) tensor
            Initial velocity field (in voxels)

        Returns
        -------
        phi : (B, D, *shape) tensor
            Displacement field (in voxels)
        """
        return super().forward(v)[1]


class Compose(nn.Module):
    """Compose two displacement fields"""

    def __init__(self, bound='circulant', order=1, backend=None):
        """
        Parameters
        ----------
        bound : [list of] {'circulant', 'neumann', 'dirichlet', 'sliding'}
            Boundary conditions.
        order : int
            Order of encoding splines.
        backend : module
            Backend to use to implement resampling.
            Must be one of the modules under `diffeo.backends`.
        """
        super().__init__()
        self.bound = bound
        self.order = order
        self.backend = backend or default_backend

    def forward(self, left, right):
        """
        Parameters
        ----------
        left : (B, D, *shape) tensor
            Displacement field (in voxels)
        right : (B, D, *shape) tensor
            Displacement field (in voxels)

        Returns
        -------
        phi : (B, D, *shape) tensor
            Composed displacment (in voxels)
        """
        left = left.movedim(1, -1)
        right = right.movedim(1, -1)
        phi = compose(
            left, right,
            bound=self.bound, order=self.order, backend=self.backend,
        )
        return phi.movedim(-1, 1)


class Pull(nn.Module):
    """Warp an image using a displacement field"""

    def __init__(self, bound='wrap', backend=None):
        """
        Parameters
        ----------
        bound : [list of] {'wrap', 'reflect', 'mirror'}
            Boundary conditions.
            If warping a displacement field, can also be one of the
            metrics bounds: {'circulant', 'neumann', 'dirichlet', 'sliding'}
        backend : module
            Backend to use to implement resampling.
            Must be one of the modules under `diffeo.backends`.
        """
        super().__init__()
        self.bound = bound
        self.backend = backend or default_backend

    def forward(self, x, phi):
        """
        Parameters
        ----------
        x : (B, C, *shape) tensor
            Image
        phi : (B, D, *shape) tensor
            Displacement field (in voxels)

        Returns
        -------
        y : (B, C, *shape) tensor
            Warped image
        """
        x = x.movedim(1, -1)
        phi = phi.movedim(1, -1)
        return self.backend.pull(x, phi, bound=self.bound).movedim(-1, 1)


class Push(nn.Module):
    """Splat an image using a displacement field"""

    def __init__(self, bound='wrap', normalize=False, backend=None):
        """
        Parameters
        ----------
        bound : [list of] {'wrap', 'reflect', 'mirror'}
            Boundary conditions.
            If splatting a displacement field, can also be one of the
            metrics bounds: {'circulant', 'neumann', 'dirichlet', 'sliding'}
        normalize : bool
            Whether to divide the pushed values with the number of
            pushed values (i.e., the "count").
        backend : module
            Backend to use to implement resampling.
            Must be one of the modules under `diffeo.backends`.
        """
        super().__init__()
        self.bound = bound
        self.normalize = normalize
        self.backend = backend or default_backend

    def forward(self, x, phi):
        """
        Parameters
        ----------
        x : (B, C, *shape) tensor
            Image
        phi : (B, D, *shape) tensor
            Displacement field (in voxels)

        Returns
        -------
        y : (B, C, *shape) tensor
            Splatted image
        """
        x = x.movedim(1, -1)
        phi = phi.movedim(1, -1)
        x = self.backend.push(x, phi, bound=self.bound)
        if self.normalize:
            c = self.backend.count(phi, bound=self.bound)
            x = x / c.clamp_min_(1e-3)
        return x.movedim(-1, 1)


class Count(nn.Module):
    """Splat an image using a displacement field"""

    def __init__(self, bound='wrap', backend=None):
        """
        Parameters
        ----------
        bound : [list of] {'wrap', 'reflect', 'mirror'}
            Boundary conditions.
            If splatting a displacement field, can also be one of the
            metrics bounds: {'circulant', 'neumann', 'dirichlet', 'sliding'}
        backend : module
            Backend to use to implement resampling.
            Must be one of the modules under `diffeo.backends`.
        """
        super().__init__()
        self.bound = bound
        self.backend = backend or default_backend

    def forward(self, phi):
        """
        Parameters
        ----------
        phi : (B, D, *shape) tensor
            Displacement field (in voxels)

        Returns
        -------
        count : (B, 1, *shape) tensor
            Count image
        """
        phi = phi.movedim(1, -1)
        return self.backend.count(phi, bound=self.bound).movedim(-1, 1)


class ToCoeff(nn.Module):
    """Compute interpolating spline coefficients"""

    def __init__(self,  bound='wrap', order=1, backend=None):
        """
        Parameters
        ----------
        bound : [list of] {'wrap', 'reflect', 'mirror'}
            Boundary conditions.
            If filtering a displacement field, can also be one of the
            metrics bounds: {'circulant', 'neumann', 'dirichlet', 'sliding'}
        order : int
            Order of encoding splines.
        backend : module
            Backend to use to implement resampling.
            Must be one of the modules under `diffeo.backends`.
        """
        super().__init__()
        self.bound = bound
        self.order = order
        self.backend = backend or default_backend

    def forward(self, x):
        """
        Parameters
        ----------
        x : (B, C, *shape) tensor
            Image

        Returns
        -------
        coeff : (B, C, *shape) tensor
            Spline coefficients
        """
        ndim = x.ndim - 2
        x = x.movedim(1, -1)
        x = self.backend.to_coeff(ndim, x, self.bound, self.order)
        return x.movedim(-1, 1)


class FromCoeff(nn.Module):
    """Interpolate spline at integer locations"""

    def __init__(self,  bound='wrap', order=1, backend=None):
        """
        Parameters
        ----------
        bound : [list of] {'wrap', 'reflect', 'mirror'}
            Boundary conditions.
            If filtering a displacement field, can also be one of the
            metrics bounds: {'circulant', 'neumann', 'dirichlet', 'sliding'}
        order : int
            Order of encoding splines.
        backend : module
            Backend to use to implement resampling.
            Must be one of the modules under `diffeo.backends`.
        """
        super().__init__()
        self.bound = bound
        self.order = order
        self.backend = backend or default_backend

    def forward(self, x):
        """
        Parameters
        ----------
        coeff : (B, C, *shape) tensor
            Spline coefficients

        Returns
        -------
        x : (B, C, *shape) tensor
            Interpolated image
        """
        ndim = x.ndim - 2
        x = x.movedim(1, -1)
        x = self.backend.from_coeff(ndim, x, self.bound, self.order)
        return x.movedim(-1, 1)


class Upsample(nn.Module):
    """Upsample an image"""

    def __init__(self, factor=2, order=1, bound='wrap', anchor='center',
                 prefilter=False, postfilter=False, backend=None):
        """
        Parameters
        ----------
        factor : [list of] int
            Upsampling factor
        order : int
            Spline interpolation order
        bound : str
            Boundary conditions
        anchor : {'center', 'edge'}
            Align either the centers or edges of the corner voxels across levels.
        prefilter : bool
            Apply spline prefiltering
            (i.e., convert input to interpolating spline coefficients)
        postfilter : bool
            Apply spline postfiltering
            (i.e., convert output to interpolating spline coefficients)
        backend : diffeo.backend
            Which interpolation backend to use.
        """
        super().__init__()
        self.factor = factor
        self.order = order
        self.bound = bound
        self.anchor = anchor
        self.prefilter = prefilter
        self.postfilter = postfilter
        self.backend = backend or default_backend

    def forward(self, x, shape=None):
        """
        Parameters
        ----------
        x : (B, C, *shape_inp) tensor
            Image
        shape : list[int], optional
            Output spatial shape

        Returns
        -------
        x : (B, C, *shape) tensor
            Upsampled image
        """
        ndim = x.ndim - 2
        x = x.movedim(1, -1)
        if self.prefilter:
            x = self.backend.to_coeff(ndim, x, self.bound, self.order)
        x = upsample(
            ndim, x,
            factor=self.factor,
            shape=shape,
            anchor=self.anchor,
            bound=self.bound,
            order=self.order,
            backend=self.backend,
        )
        if self.postfilter:
            x = self.backend.to_coeff(ndim, x, self.bound, self.order)
        x = x.movedim(-1, 1)
        return x


class Downsample(nn.Module):
    """Downsample an image"""

    def __init__(self, factor=2, order=1, bound='wrap', anchor='center',
                 prefilter=False, postfilter=False, backend=None):
        """
        Parameters
        ----------
        factor : [list of] int
            Downsampling factor
        order : int
            Spline interpolation order
        bound : str
            Boundary conditions
        anchor : {'center', 'edge'}
            Align either the centers or edges of the corner voxels across levels.
        prefilter : bool
            Apply spline prefiltering
            (i.e., convert input to interpolating spline coefficients)
        postfilter : bool
            Apply spline postfiltering
            (i.e., convert output to interpolating spline coefficients)
        backend : diffeo.backend
            Which interpolation backend to use.
        """
        super().__init__()
        self.factor = factor
        self.order = order
        self.bound = bound
        self.anchor = anchor
        self.prefilter = prefilter
        self.postfilter = postfilter
        self.backend = backend or default_backend

    def forward(self, x, shape=None):
        """
        Parameters
        ----------
        x : (B, C, *shape_inp) tensor
            Image
        shape : list[int], optional
            Output spatial shape

        Returns
        -------
        x : (B, C, *shape) tensor
            Upsampled image
        """
        ndim = x.ndim - 2
        x = x.movedim(1, -1)
        if self.prefilter:
            x = self.backend.to_coeff(ndim, x, self.bound, self.order)
        x = downsample(
            ndim, x,
            factor=self.factor,
            shape=shape,
            anchor=self.anchor,
            bound=self.bound,
            order=self.order,
            backend=self.backend,
        )
        if self.postfilter:
            x = self.backend.to_coeff(ndim, x, self.bound, self.order)
        x = x.movedim(-1, 1)
        return x


class UpsampleFlow(nn.Module):
    """Upsample a displacement field"""

    def __init__(self, factor=2, order=1, bound='circulant', anchor='center',
                 prefilter=False, postfilter=False, backend=None):
        """
        Parameters
        ----------
        factor : [list of] int
            Upsampling factor
        order : int
            Spline interpolation order
        bound : str
            Boundary conditions
        anchor : {'center', 'edge'}
            Align either the centers or edges of the corner voxels across levels.
        prefilter : bool
            Apply spline prefiltering
            (i.e., convert input to interpolating spline coefficients)
        postfilter : bool
            Apply spline postfiltering
            (i.e., convert output to interpolating spline coefficients)
        backend : diffeo.backend
            Which interpolation backend to use.
        """
        super().__init__()
        self.factor = factor
        self.order = order
        self.bound = bound
        self.anchor = anchor
        self.prefilter = prefilter
        self.postfilter = postfilter
        self.backend = backend or default_backend

    def forward(self, x, shape=None):
        """
        Parameters
        ----------
        x : (B, C, *shape_inp) tensor
            Image
        shape : list[int], optional
            Output spatial shape

        Returns
        -------
        x : (B, C, *shape) tensor
            Upsampled image
        """
        ndim = x.ndim - 2
        x = x.movedim(1, -1)
        if self.prefilter:
            x = self.backend.to_coeff(ndim, x, self.bound, self.order)
        x = upsample_flow(
            x,
            factor=self.factor,
            shape=shape,
            anchor=self.anchor,
            bound=self.bound,
            order=self.order,
            backend=self.backend,
        )
        if self.postfilter:
            x = self.backend.to_coeff(ndim, x, self.bound, self.order)
        x = x.movedim(-1, 1)
        return x


class DownsampleFlow(nn.Module):
    """Downsample a displacement field"""

    def __init__(self, factor=2, order=1, bound='circulant', anchor='center',
                 prefilter=False, postfilter=False, backend=None):
        """
        Parameters
        ----------
        factor : [list of] int
            Downsampling factor
        order : int
            Spline interpolation order
        bound : str
            Boundary conditions
        anchor : {'center', 'edge'}
            Align either the centers or edges of the corner voxels across levels.
        prefilter : bool
            Apply spline prefiltering
            (i.e., convert input to interpolating spline coefficients)
        postfilter : bool
            Apply spline postfiltering
            (i.e., convert output to interpolating spline coefficients)
        backend : diffeo.backend
            Which interpolation backend to use.
        """
        super().__init__()
        self.factor = factor
        self.order = order
        self.bound = bound
        self.anchor = anchor
        self.prefilter = prefilter
        self.postfilter = postfilter
        self.backend = backend or default_backend

    def forward(self, x, shape=None):
        """
        Parameters
        ----------
        x : (B, C, *shape_inp) tensor
            Image
        shape : list[int], optional
            Output spatial shape

        Returns
        -------
        x : (B, C, *shape) tensor
            Upsampled image
        """
        ndim = x.ndim - 2
        x = x.movedim(1, -1)
        if self.prefilter:
            x = self.backend.to_coeff(ndim, x, self.bound, self.order)
        x = downsample_flow(
            x,
            factor=self.factor,
            shape=shape,
            anchor=self.anchor,
            bound=self.bound,
            order=self.order,
            backend=self.backend,
        )
        if self.postfilter:
            x = self.backend.to_coeff(ndim, x, self.bound, self.order)
        x = x.movedim(-1, 1)
        return x
