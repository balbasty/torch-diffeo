from torch import nn
from diffeo.svf import exp, bch
from diffeo.shoot import shoot, default_metric
from diffeo.backends import default_backend
from diffeo.flows import compose


class Exp(nn.Module):
    """Exponentiate a Stationary Velocity Field"""

    def __init__(self, bound='dft', steps=8, anagrad=False, backend=None):
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
    """Compose two Stationary Velocity Fields using the BCH formula"""

    def __init__(self, bound='dft', order=2, backend=None):
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
    """Exponentiate an Initial Velocity using Geodesic Shooting"""

    def __init__(self, metric=default_metric, steps=8, fast=True,
                 backend=None):
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
    """Exponentiate an Initial Velocity using Geodesic Shooting"""

    def __init__(self, metric=default_metric, steps=8, fast=True,
                 backend=None):
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
    """Exponentiate an Initial Velocity using Geodesic Shooting"""

    def __init__(self, metric=default_metric, steps=8, fast=True,
                 backend=None):
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

    def __init__(self, bound='dft', backend=None):
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

    def __init__(self, bound='dft', backend=None):
        super().__init__()
        self.bound = bound
        self.backend = backend or default_backend

    def forward(self, x, phi):
        x = x.movedim(1, -1)
        phi = phi.movedim(1, -1)
        return self.backend.pull(x, phi, bound=self.bound).movedim(-1, 1)


class Push(nn.Module):
    """Splat an image using a displacement field"""

    def __init__(self, bound='dft', normalize=False, backend=None):
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

    def __init__(self, bound='dft', backend=None):
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