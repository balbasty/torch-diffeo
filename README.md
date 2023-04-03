# torch-diffeo
Scaling-and-squaring and Geodesic Shooting layers in PyTorch

## Layers

```python
Exp(bound='dft', steps=8, anagrad=False): ...
"""Exponentiate a Stationary Velocity Field

Parameters
----------
bound : {'dft', 'dct1', 'dct2', 'dst1', 'dst2'}
    Boundary conditions.
steps : int
    Number of scaling and squaring steps.
anagrad : bool
    Use analytical gradients instead of autograd.
"""

BCH(bound='dft', order=2): ...
"""Compose two Stationary Velocity Fields using the BCH formula

The Baker–Campbell–Hausdorff (BCH) allows computing z such that
exp(z) = exp(x) o exp(y).

https://en.wikipedia.org/wiki/BCH_formula
    
Parameters
----------
bound : {'dft', 'dct1', 'dct2', 'dst1', 'dst2'}
    Boundary conditions.
order : int
    Maximum order used in the BCH series
"""

Shoot(metric=Mixture(), steps=8, fast=True): ...
ShootInv(metric=Mixture(), steps=8, fast=True): ...
ShootBoth(metric=Mixture(), steps=8, fast=True): ...
"""Exponentiate an Initial Velocity Field by Geodesic Shooting

Parameters
----------
metric : Metric
    A Riemannian metric
steps : int
    Number of Euler integration steps.
fast : int
    Use a faster but slightly less accurate integration scheme.
"""

Compose(bound='dft'): ...
"""Compose two Displacement Fields

Parameters
----------
bound : {'dft', 'dct1', 'dct2', 'dst1', 'dst2'}
    Boundary conditions.
"""

Pull(bound='dft'): ...
"""Warp an image using a Displacement Field

Parameters
----------
bound : {'dft', 'dct1', 'dct2', 'dst1', 'dst2'}
    Boundary conditions.
"""

Push(bound='dft', normalize=False): ...
"""Splat an image using a Displacement Field

Parameters
----------
bound : {'dft', 'dct1', 'dct2', 'dst1', 'dst2'}
    Boundary conditions.
normalize : bool
    Divide the pushed values by the result of `Count`.
"""

Count(bound='dft'): ...
"""Splat an image of ones using a Displacement Field

Parameters
----------
bound : {'dft', 'dct1', 'dct2', 'dst1', 'dst2'}
    Boundary conditions.
"""
```

## Metrics

We define a range of Riemannian metrics that can be used to regularize
velocity fields, and must be used for Geodesic Shooting.

All metrics implement the following methods:
```python
metric.forward(v: Tensor) -> Tensor: ...
"""
Apply the forward linear operator `L`

Parameters
----------
v : (..., *spatial, D) tensor
    A velocity field.

Returns
-------
m : (..., *spatial, D) tensor
    A momentum field.
"""

metric.inverse(m: Tensor) -> Tensor: ...
"""
Apply the inverse linear operator `K = inv(L)`

Parameters
----------
m : (..., *spatial, D) tensor
    A momentum field.

Returns
-------
v : (..., *spatial, D) tensor
    A velocity field.
"""

metric.whiten(v: Tensor) -> Tensor: ...
"""
Apply the square root of the inverse linear operator `sqrt(K)`

Parameters
----------
v : (..., *spatial, D) tensor
    A velocity field.
    
Returns
-------
x : (..., *spatial, D) tensor
    A white field.
"""

metric.color(x: Tensor) -> Tensor: ...
"""
Apply the square root of the linear operator `sqrt(L)`

Parameters
----------
x : (..., *spatial, D) tensor
    A white field.

Returns
-------
v : (..., *spatial, D) tensor
    A velocity field.
"""

metric.logdet(v: Tensor) -> Tensor: ...
"""
Compute the log-determinant of the linear operator `logdet(L)`

Parameters
----------
v : (..., *spatial, D) tensor
    A velocity field. 
    Its values are not used. Only its shape, dtype and device are used.
    
Returns
-------
ld : scalar tensor
    Log-determinant (scaled by batch size).
"""
```

This is the list metrics currently available:
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

Laplace(factor=1, voxel_size=1, bound='dft',
        learnable=False, cache=False): ...
"""
Positive semi-definite metric based on the Laplace operator.
This is relatively similar to SPM's "membrane" energy, but relies on
the (ill-posed) analytical form of the Greens function.

https://en.wikipedia.org/wiki/Laplace%27s_equation
https://en.wikipedia.org/wiki/Green%27s_function

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

Helmoltz(factor=1, alpha=1e-3, voxel_size=1, bound='dft',
         learnable=False, cache=False): ...
"""
Positive semi-definite metric based on the Helmoltz operator.
This is relatively similar to SPM's mixture of "absolute" and
"membrane" energies, but relies on the (ill-posed) analytical form
of the Greens function.

https://en.wikipedia.org/wiki/Helmholtz_equation
https://en.wikipedia.org/wiki/Green%27s_function

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

Gaussian(fwhm=16, factor=1, voxel_size=1, bound='dft',
         learnable=False, cache=False): ...
"""
Positive semi-definite metric whose Greens function is a Gaussian filter.

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
    This cannot be ]()used when `learnable='fwhm'`
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
  **This is the default backend.**
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

Note that we currently have issues when using the `torch` backend along
with geodesic shooting layers. Classic interpolation and stationary 
velocity fields should work fine, however.
