# torch-diffeo
Scaling-and-squaring and Geodesic Shooting layers in PyTorch

## Getting started

This package requires `pytorch >= 1.8` and 
[`torch-interpol`](https://github.com/balbasty/torch-interpol). 
We require this version so that pytorch supports complex values and the
modern `torch.fft` module. To install with pip, simply do:
```shell
pip install "torch-diffeo @ git+https://github.com/balbasty/torch-diffeo"
```
However, it is in general advised to install pytorch using `conda`
to minimizing conflicts:
```shell
conda install -c pytorch -c nvidia pytorch cudatoolkit=10.2
pip install "torch-diffeo @ git+https://github.com/balbasty/torch-diffeo"
```

## Layers

```python
Exp(bound='circulant', order=1, steps=8, anagrad=False): ...
ExpInv(bound='circulant', order=1, steps=8, anagrad=False): ...
ExpBoth(bound='circulant', order=1, steps=8, anagrad=False): ...
"""Exponentiate a Stationary Velocity Field

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
"""

BCH(bound='circulant', trunc=2, order=1): ...
"""Compose two Stationary Velocity Fields using the BCH formula

The Baker–Campbell–Hausdorff (BCH) allows computing z such that
exp(z) = exp(x) o exp(y).

https://en.wikipedia.org/wiki/BCH_formula
    
Parameters
----------
bound : [list of] {'circulant', 'neumann', 'dirichlet', 'sliding'}
    Boundary conditions.
trunc : int
    Maximum order used in the BCH series
order : int
    Order of encoding splines
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

Compose(bound='circulant', order=1): ...
"""Compose two Displacement Fields

Parameters
----------
bound : [list of] {'circulant', 'neumann', 'dirichlet', 'sliding'}
    Boundary conditions.
order : int
    Order of encoding splines
"""

Pull(bound='wrap'): ...
"""Warp an image using a Displacement Field

Parameters
----------
bound : [list of] {'wrap', 'reflect', 'mirror'} 
    Boundary conditions.
    If splatting a displacement field, can also be one of the 
    metrics bounds: {'circulant', 'neumann', 'dirichlet', 'sliding'}
"""

Push(bound='wrap', normalize=False): ...
"""Splat an image using a Displacement Field

Parameters
----------
bound : [list of] {'wrap', 'reflect', 'mirror'} 
    Boundary conditions.
    If splatting a displacement field, can also be one of the 
    metrics bounds: {'circulant', 'neumann', 'dirichlet', 'sliding'}
normalize : bool
    Divide the pushed values by the result of `Count`.
"""

Count(bound='wrap'): ...
"""Splat an image of ones using a Displacement Field

Parameters
----------
bound : [list of] {'wrap', 'reflect', 'mirror'} 
    Boundary conditions.
    If splatting a displacement field, can also be one of the 
    metrics bounds: {'circulant', 'neumann', 'dirichlet', 'sliding'}
"""

ToCoeff(bound='wrap', order=1): ...
"""Compute interpolating spline coefficient

Parameters
----------
bound : [list of] {'wrap', 'reflect', 'mirror'}
    Boundary conditions.
    If filtering a displacement field, can also be one of the
    metrics bounds: {'circulant', 'neumann', 'dirichlet', 'sliding'}
"""

FromCoeff(bound='wrap', order=1): ...
"""Interpolate spline at integer locations

Parameters
----------
bound : [list of] {'wrap', 'reflect', 'mirror'}
    Boundary conditions.
    If filtering a displacement field, can also be one of the
    metrics bounds: {'circulant', 'neumann', 'dirichlet', 'sliding'}
"""

Upsample(factor=2, order=1, bound='wrap', anchor='center',
         prefilter=False, postfilter=False): ...
"""
Upsample an image

Parameters
----------
factor : [list of] int
    Upsampling factor
order : int
    Spline interpolation order
bound : [list of] {'wrap', 'reflect', 'mirror'}
    Boundary conditions.
anchor : {'center', 'edge'}
    Align either the centers or edges of the corner voxels across levels.
prefilter : bool
    Apply spline prefiltering
    (i.e., convert input to interpolating spline coefficients)
postfilter : bool
    Apply spline postfiltering
    (i.e., convert output to interpolating spline coefficients)
"""

Downsample(factor=2, order=1, bound='wrap', anchor='center',
           prefilter=False, postfilter=False): ...
"""
Downsample an image

Parameters
----------
factor : [list of] int
    Downsampling factor
order : int
    Spline interpolation order
bound : [list of] {'wrap', 'reflect', 'mirror'}
    Boundary conditions.
anchor : {'center', 'edge'}
    Align either the centers or edges of the corner voxels across levels.
prefilter : bool
    Apply spline prefiltering
    (i.e., convert input to interpolating spline coefficients)
postfilter : bool
    Apply spline postfiltering
    (i.e., convert output to interpolating spline coefficients)
"""

UpsampleFlow(factor=2, order=1, bound='circulant', anchor='center',
             prefilter=False, postfilter=False): ...
"""
Upsample a displacement field

Parameters
----------
factor : [list of] int
    Upsampling factor
order : int
    Spline interpolation order
bound : [list of] {'circulant', 'neumann', 'dirichlet', 'sliding'}
    Boundary conditions.
anchor : {'center', 'edge'}
    Align either the centers or edges of the corner voxels across levels.
prefilter : bool
    Apply spline prefiltering
    (i.e., convert input to interpolating spline coefficients)
postfilter : bool
    Apply spline postfiltering
    (i.e., convert output to interpolating spline coefficients)
"""

DownsampleFlow(factor=2, order=1, bound='circulant', anchor='center',
               prefilter=False, postfilter=False): ...
"""
Downsample a displacement field

Parameters
----------
factor : [list of] int
    Downsampling factor
order : int
    Spline interpolation order
bound : [list of] {'circulant', 'neumann', 'dirichlet', 'sliding'}
    Boundary conditions.
anchor : {'center', 'edge'}
    Align either the centers or edges of the corner voxels across levels.
prefilter : bool
    Apply spline prefiltering
    (i.e., convert input to interpolating spline coefficients)
postfilter : bool
    Apply spline postfiltering
    (i.e., convert output to interpolating spline coefficients)
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
        factor=1, voxel_size=1, bound='circulant', use_diff=True,
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
bound : [list of] {'circulant', 'neumann', 'dirichlet', 'sliding'}
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

SplineMixture(absolute=0, membrane=0, bending=0, lame_shears=0, lame_div=0,
              factor=1, voxel_size=1, bound='circulant', order=3, use_conv=True,
              learnable=False, cache=False): ...
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

Laplace(factor=1, voxel_size=1, bound='circulant',
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
bound : [list of] {'circulant', 'neumann', 'dirichlet', 'sliding'}
    Boundary conditions
learnable : bool
    Make `factor` a learnable parameter
cache : bool or int
    Cache up to `n` kernels
"""

Helmoltz(factor=1, alpha=1e-3, voxel_size=1, bound='circulant',
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
bound : [list of] {'circulant', 'neumann', 'dirichlet', 'sliding'}
    Boundary conditions
learnable : bool
    Make `factor` a learnable parameter
cache : bool or int
    Cache up to `n` kernels
"""

Gaussian(fwhm=16, factor=1, voxel_size=1, bound='circulant',
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
bound : [list of] {'circulant', 'neumann', 'dirichlet', 'sliding'}
    Boundary conditions
learnable : bool or {'factor', 'fwhm', 'fwhm+factor}
    Make `factor` and/or 'fwhm' a learnable parameter.
    `True` is equivalent to `factor`.
cache : bool or int
    Cache up to `n` kernels
    This cannot be used when `learnable='fwhm'`
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

All backends implement the following function:
```python
def pull(image, flow, bound='dct2', has_identity=False): ...
"""Warp an image according to a (voxel) displacement field.

Parameters
----------
image : (..., *shape_in, C) tensor
    Input image.
flow : (..., *shape_out, D) tensor
    Displacement field, in voxels.
bound : {'dft', 'dct{1|2|3|4}', 'dst{1|2|3|4}'}, default='dct2'
    Boundary conditions.
    Can also be one for {'circulant', 'neumann', 'dirichlet', 'sliding'},
    in which case the image is assumed to be a flow field.
has_identity : bool, default=False
    - If False, `flow` is contains relative displacement.
    - If True, `flow` contains absolute coordinates.

Returns
-------
warped : (..., *shape_out, C) tensor
    Warped image
"""

def push(image, flow, shape=None, bound='dct2', has_identity=False): ...
"""Splat an image according to a (voxel) displacement field.

Parameters
----------
image : (..., *shape_in, C) tensor
    Input image.
flow : (..., *shape_out, D) tensor
    Displacement field, in voxels.
shape : list[int], optional
    Output shape
bound : {'dft', 'dct{1|2|3|4}', 'dst{1|2|3|4}'}, default='dct2'
    Boundary conditions.
    Can also be one for {'circulant', 'neumann', 'dirichlet', 'sliding'},
    in which case the image is assumed to be a flow field.
has_identity : bool, default=False
    - If False, `flow` is contains relative displacement.
    - If True, `flow` contains absolute coordinates.

Returns
-------
pushed : (..., *shape_out, C) tensor
    Pushed image

"""

def count(flow, shape=None, bound='dct2', has_identity=False): ...
"""Splat an image of ones according to a (voxel) displacement field.

Parameters
----------
flow : (..., *shape_out, D) tensor
    Displacement field, in voxels.
shape : list[int], optional
    Output shape
bound : {'dft', 'dct{1|2|3|4}', 'dst{1|2|3|4}'}, default='dct2'
    Boundary conditions.
    Can also be one for {'circulant', 'neumann', 'dirichlet', 'sliding'},
    in which case the count image may have D channels.
has_identity : bool, default=False
    - If False, `flow` is contains relative displacement.
    - If True, `flow` contains absolute coordinates.

Returns
-------
count : (..., *shape_out, 1|D) tensor
    Count image
"""

def grad(image, flow, bound='dct2', has_identity=False): ...
"""Compute spatial gradients of image according to a (voxel) displacement field.

Parameters
----------
image : (..., *shape_in, C) tensor
    Input image.
flow : (..., *shape_out, D) tensor
    Displacement field, in voxels.
bound : {'dft', 'dct{1|2|3|4}', 'dst{1|2|3|4}'}, default='dct2'
    Boundary conditions.
    Can also be one for {'circulant', 'neumann', 'dirichlet', 'sliding'},
    in which case the image is assumed to be a flow field.
has_identity : bool, default=False
    - If False, `flow` is contains relative displacement.
    - If True, `flow` contains absolute coordinates.

Returns
-------
grad : (..., *shape_out, C, D) tensor
    Sampled gradients
"""
```
