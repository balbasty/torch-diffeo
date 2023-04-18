"""
Boundary conditions
===================

There is no common convention to name boundary conditions.
Here's the list all possible aliases.

=========   ===========   =======================   =======================
Fourier     SciPy         Metric                    Description
=========   ===========   =======================   =======================
dft         wrap          circular                  c  d | a b c d |  a  b
dct2        reflect       neumann                   b  a | a b c d |  d  c
dct1        mirror                                  c  b | a b c d |  c  b
dst2                      dirichlet                -b -a | a b c d | -d -c
dst1                                               -a  0 | a b c d |  0 -d

We further define a flow-specific "sliding" boundary condition, which
uses a combination of dct2 and dst2:
=============================   =============================
X component                     Y component
=============================   =============================
 -f -e   -e -f -g -h   -h -g     -f -e    e  f  g  h   -h -g
 -b -a   -a -b -c -d   -d -c     -b -a    a  b  c  d   -d -c
-----------------------------   -----------------------------
  b  a |  a  b  c  d |  d  c     -b -a |  a  b  c  d | -d -c
  f  e |  e  f  g  h |  h  g     -f -e |  e  f  g  h | -h -g
  j  i |  i  j  k  l |  l  k     -j -i |  i  j  k  l | -l -k
  l  m |  m  n  o  p |  p  o     -n -m |  m  n  o  p | -p -o
-----------------------------   -----------------------------
 -n -m   -m -n -o -p   -p -o     -n -m    m  n  o  p   -p -o
 -j -i   -i -j -k -l   -l -k     -j -i    i  j  k  l   -l -k
"""
from torch import nn
from diffeo.svf import exp, bch
from diffeo.shoot import shoot, default_metric
from diffeo.backends import default_backend
from diffeo.flows import compose


class Exp(nn.Module):
    """Exponentiate a Stationary Velocity Field"""

    def __init__(self, bound='circulant', steps=8, anagrad=False, backend=None):
        """
        Parameters
        ----------
        bound : [list of] {'circulant', 'neumann', 'dirichlet', 'sliding'}
            Boundary conditions.
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
        self.steps = steps
        self.anagrad = anagrad
        self.backend = backend or default_backend

    def forward(self, v):
        v = v.movedim(1, -1)
        v = exp(v, self.steps, self.bound, self.anagrad, self.backend)
        v = v.movedim(-1, 1)
        return v


class BCH(nn.Module):
    """
    Compose two Stationary Velocity Fields using the BCH formula

    The Baker–Campbell–Hausdorff (BCH) allows computing z such that
    exp(z) = exp(x) o exp(y).

    https://en.wikipedia.org/wiki/BCH_formula
    """

    def __init__(self, bound='circulant', order=2, backend=None):
        """
        Parameters
        ----------
        bound : [list of] {'circulant', 'neumann', 'dirichlet', 'sliding'}
            Boundary conditions.
        order : int
            Maximum order used in the BCH series
        backend : module
            Backend to use to implement resampling.
            Must be one of the modules under `diffeo.backends`.
        """
        super().__init__()
        self.bound = bound
        self.order = order
        self.backend = backend or default_backend

    def forward(self, left, right):
        left = left.movedim(1, -1)
        right = right.movedim(1, -1)
        v = bch(left, right, self.order, self.bound, self.backend)
        v = v.movedim(-1, 1)
        return v


class Shoot(nn.Module):
    """
    Exponentiate an Initial Velocity using Geodesic Shooting

    Returns the forward transform.
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
        v = v.movedim(1, -1)
        phi, _ = shoot(v, self.metric, self.steps, self.fast, backend=self.backend)
        return phi.movedim(-1, 1)


class ShootInv(nn.Module):
    """
    Exponentiate an Initial Velocity using Geodesic Shooting

    Returns the inverse transform.
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
        v = v.movedim(1, -1)
        _, iphi = shoot(v, self.metric, self.steps, self.fast,
                        backend=self.backend)
        return iphi.movedim(-1, 1)


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
        phi, iphi = shoot(v.movedim(1, -1), self.metric, self.steps, self.fast,
                          backend=self.backend)
        return phi.movedim(-1, 1), iphi.movedim(-1, 1)


class Compose(nn.Module):
    """Compose two displacement fields"""

    def __init__(self, bound='circulant', backend=None):
        """
        Parameters
        ----------
        bound : [list of] {'circulant', 'neumann', 'dirichlet', 'sliding'}
            Boundary conditions.
        backend : module
            Backend to use to implement resampling.
            Must be one of the modules under `diffeo.backends`.
        """
        super().__init__()
        self.bound = bound
        self.backend = backend or default_backend

    def forward(self, left, right):
        left = left.movedim(1, -1)
        right = right.movedim(1, -1)
        phi = compose(left, right, bound=self.bound, backend=self.backend)
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

    def forward(self, x, phi):
        x = x.movedim(1, -1)
        phi = phi.movedim(1, -1)
        return self.backend.count(x, phi, bound=self.bound).movedim(-1, 1)


class Upsample(nn.Module):
    pass


class Downsample(nn.Module):
    pass
