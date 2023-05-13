import torch.fft
from torch import Tensor
from math import floor
from typing import Union, List, Optional
from diffeo.utils import ensure_list
from . import alltransforms


def dtn(
    x: Tensor,
    type: Union[str, List[str]] = 'dft',
    inverse: bool = False,
    dim: Optional[Union[int, List[int]]] = None,
    norm: Optional[str] = "backward",
    force_real: bool = False,
) -> Tensor:
    """Compute a multidimensional Discrete Transform, different per dimension.

    Parameters
    ----------
    x : torch.tensor
        The input array.
    type : [list of] {"dft", "dct{1234}", "dst{1234}"}, optional
        Type of transform, per dimension. Default is "dft".
    inverse : bool, optional
        Apply inverse transform
    dim : [list of] int, optional
        Axes over which the transform is computed.
        If not given, all axes are used.
    norm : [list of] {"backward", "ortho", "forward"}, optional
        Normalization mode. Default is "backward".

    Returns
    -------
    y : torch.tensor
        The transformed input array.
    """
    # same type in all dimensions
    if isinstance(type, str):
        if inverse:
            type = 'i' + type
        dt = getattr(alltransforms, type)
        x = dt(x, dim=dim, norm=norm)
        if force_real:
            x = real(x)
        return x

    # different types across dimensions
    if dim is None:
        dim = list(range(x.ndim))
    dim = ensure_list(dim, len(dim))
    type = ensure_list(type, len(dim))
    norm = ensure_list(norm, len(dim))
    if len(dim) == 0:
        return x

    for t, n, d in zip(type, norm, dim):
        if inverse:
            t = 'i' + t
        dt = getattr(alltransforms, t)
        x = dt(x, dim=d, norm=n)
        if force_real:
            x = real(x)
    return x


def kerdtn(
    kernel: Tensor,
    ndim: int,
    sym: Union[bool, List[bool]] = False,
    inverse: bool = False,
    norm: Optional[str] = "backward",
) -> Tensor:
    """Kernel discrete transform

    Parameters
    ----------
    kernel : ([D], D, *shape) tensor
        Input kernel.
    ndim : int
        Number of spatial dimensions
    sym : [list of] bool
        Whether to apply a symmetric transform (DCT/DST), per dimension.
    inverse : bool
        Whether to apply the inverse transform

    Returns
    -------
    kernel : ([D], D, *shape) tensor
        Transformed kernel

    """
    dims = list(range(-ndim, 0))
    sym = ensure_list(sym, ndim)
    if not any(sym):
        dt = alltransforms.idft if inverse else alltransforms.dft
        return real(dt(kernel, dim=dims))

    dbound = ['dct1' if is_sym else 'dft' for is_sym in sym]
    obound = ['dst1' if is_sym else 'dft' for is_sym in sym]

    def apply(x, bound):
        return dtn(x, bound, dim=dims, inverse=inverse, force_real=True, norm=norm)

    nbatch = kernel.ndim - ndim
    if nbatch == 2:
        kernel, y = torch.zeros_like(kernel), kernel
        for d, m in enumerate(sym):
            kernel[d, d] = apply(y[d, d], dbound)
            for dd in range(d):
                kernel[d, dd] = apply(y[d, dd], obound)
            for dd in range(d+1, ndim):
                kernel[d, dd] = apply(y[d, dd], obound)
    else:
        kernel = apply(kernel, dbound)
    return kernel


def dtshift(
    kernel: Tensor,
    ndim: int,
    sym: Union[bool, List[bool]] = False,
) -> Tensor:
    """Shift a potentially symmetric kernel

    Parameters
    ----------
    kernel : ([D], D, *shape) tensor
        Input kernel.
        If it has a single batch dimension, it is a diagonal kernel.
        Otherwise, it is a full kernel. If `sym`, diagonal element are
        assumed to have an odd symmetry (DCT-I) and off-diagonal elements
        are assumed to have an odd antisymmetry (DST-I).
    ndim : int
        Number of spatial dimensions.
    sym : [list of] bool
        Whether to apply a symmetric transform (DCT/DST), per dimension.

    Returns
    -------
    kernel : ([D], D, *shape) tensor
        Shifted kernel

    """
    sym = ensure_list(sym, ndim)
    if not any(sym):
        return torch.fft.ifftshift(kernel, list(range(-ndim, 0)))

    # only keep the lower half part of the kernel
    # the rest can be reconstructed by symmetry and is
    # not needed by DCT
    #
    # in the linear-elastic case, the off-diagonal elements
    # of the kernel have an odd-symmetry with zero center
    # and the symmetric part must therefore be shifted by
    # one voxel

    nbatch = kernel.ndim - ndim
    shape = kernel.shape[-ndim:]
    tmp, kernel = kernel, torch.zeros_like(kernel)
    slicer = [slice(int(floor(s / 2)), None)  if is_sym else slice(None)
              for s, is_sym in zip(shape, sym)]
    if nbatch == 2:
        slicer0 = [slice(int(floor(s / 2)), -1) if is_sym else slice(None)
                   for s, is_sym in zip(shape, sym)]
        slicer1 = [slice(int(floor(s / 2)) + 1, None) if is_sym else slice(None)
                   for s, is_sym in zip(shape, sym)]
        for d, is_sym in enumerate(sym):
            kernel[(d, d, *slicer)] = tmp[(d, d, *slicer)]
            for dd in range(d):
                kernel[(d, dd, *slicer0)] = tmp[(d, dd, *slicer1)]
                kernel[(dd, d, *slicer0)] = tmp[(dd, d, *slicer1)]
    else:
        slicer = (slice(None),) * nbatch + tuple(slicer)
        kernel[slicer] = tmp[slicer]
    kernel = torch.fft.ifftshift(kernel, list(range(-ndim, 0)))
    return kernel


def real(x: Tensor) -> Tensor:
    """Return the real part (even if already real)"""
    if x.is_complex():
        x = x.real
    return x
