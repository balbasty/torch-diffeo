# torch-diffeo
Scaling-and-squaring and Geodesic Shooting layers in PyTorch

## Layers

```python
Exp(bound='dft', steps=8, anagrad=False): ...
"""Exponentiate a Stationary Velocity Field"""

BCH(bound='dft', order=2): ...
"""Compose two Stationary Velocity Fields using the BCH formula"""

Shoot(metric=Mixture(), steps=8, fast=True): ...
ShootInv(metric=Mixture(), steps=8, fast=True): ...
ShootBoth(metric=Mixture(), steps=8, fast=True): ...
"""Exponentiate an Initial Velocity Field by Geodesic Shooting"""

Compose(bound='dft'): ...
"""Compose two Displacement Fields"""

Pull(bound='dft'): ...
"""Warp an image using a Displacement Field"""

Push(bound='dft', normalize=False): ...
"""Splat an image using a Displacement Field"""

Count(bound='dft'): ...
"""Splat an image of ones using a Displacement Field"""
```

## Metrics

We define a range of Riemannian metrics that can be used to regularize
velocity fields, and must be used for Geodesic Shooting.

```python
Mixture(absolute=0, membrane=0, bending=0, lame_shears=0, lame_div=0,
        factor=1, voxel_size=1, bound='dft', use_diff=True,
        learnable=False, cache=False): ...
"""
Positive semi-definite metric based on finite-difference regularisers.

Mixture of "absolute", "membrane", "bending" and "linear-elastic" energies.
Note that these quantities refer to what's penalised when computing the
inner product (v, Lv). The "membrane" energy is therefore closely related
to the "Laplacian" metric.
"""

Laplace(factor=1, voxel_size=1, bound='dft',
        learnable=False, cache=False): ...
"""
Positive semi-definite metric based on the Laplace operator.
This is relatively similar to SPM's "membrane" energy, but relies on
the (ill-posed) analytical form of the Greens function.

https://en.wikipedia.org/wiki/Laplace%27s_equation
https://en.wikipedia.org/wiki/Green%27s_function
"""

Helmoltz(factor=1, alpha=1e-3, voxel_size=1, bound='dft',
         learnable=False, cache=False): ...
"""
Positive semi-definite metric based on the Helmoltz operator.
This is relatively similar to SPM's mixture of "absolute" and
"membrane" energies, but relies on the (ill-posed) analytical form
of the Greens function.

https://en.wikipedia.org/wiki/Helmholtz_equation
https://en.wikipedia.org/wiki/Green%27s_function
"""

Gaussian(fwhm=16, factor=1, voxel_size=1, bound='dft',
         earnable=False, cache=False): ...
"""
Positive semi-definite metric whose Greens function is a Gaussian filter.
"""
```

## Backends

We handle three different backends for performing the underlying sampling
operations:

- `torch`: This backend uses `torch.grid_sample`. It does not implement all
  the boundary conditions that are handled by our metric, and uses a very
  approximate implementation of splatting. It should be fast, but also
  quite inaccurate.
- `interpol`: This backend uses the package
  [`torch-interpol`](https://github.com/balbasty/torch-interpol), which
  implements all the necessary operators using TorchScript. It is not the
  fastest but all operators and boundary conditions should be consistent.
- `jitfields`: This backend uses the package
  [`jitfields`](https://github.com/balbasty/jitfields), which
  implements the same operators as `torch-interpol`, but in pure C++/CUDA.
  It does require additional dependencies (`cupy` and `cppyy`), though.
  Therefore, `jitfields` is not a mandatory dependency of `torch-diffeo`
  and must be manually intstalled by the user.

All our layers and functions take a `backend` argument:
```python
from diffeo.layers import Exp
from diffeo.backends import jitfields

layer = Exp(backend=jitfields)
```
Alternatively, we provide a context manager that sets the backend for
an entire block:
```python
from diffeo.layers import Exp, BCH
from diffeo.backends import backend, jitfields

with backend(jitfields):
    layer1 = Exp()
    layer2 = BCH()
```

