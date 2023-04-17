import torch.fft
from diffeo.utils import ensure_list
from . import alltransforms
from math import floor


def dtn(x, type='dft', inverse=False, dim=None, norm=None):
    """Compute a multidimensional Discrete Transform, different per dimension.

    Parameters
    ----------
    x : torch.tensor
        The input array.
    type : [list of] {"dft", "dct{1234}", "dst{1234}"}, optional
        Type of transform. Default is "dft".
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
    return x


def kerdtn(kernel, ndim, sym=False, inverse=False):
    """Kernel discrete transform

    Parameters
    ----------
    kernel : ([D], D, *shape) tensor
        Input kernel.
    ndim : int
        Number of spatial dimensions
    sym : bool
        Use a symmetric tranform
    inverse : bool
        Whether to apply the inverse transform

    Returns
    -------
    kernel : ([D], D, *shape) tensor
        Transformed kernel

    """
    dims = list(range(-ndim, 0))
    if not sym:
        dt = alltransforms.idft if inverse else alltransforms.dft
        return real(dt(kernel, dim=dims))

    dct = alltransforms.idct1 if inverse else alltransforms.dct1
    dst = alltransforms.idst1 if inverse else alltransforms.dst1

    nbatch = kernel.ndim - ndim
    if nbatch == 2:
        kernel, y = torch.zeros_like(kernel), kernel
        for d in range(ndim):
            kernel[d, d] = dct(y[d, d], dim=dims)
            for dd in range(d):
                kernel[d, dd] = dst(y[d, dd], dim=dims)
            for dd in range(d+1, ndim):
                kernel[d, dd] = dst(y[d, dd], dim=dims)
    else:
        kernel = dct(kernel, dim=dims)
    return kernel


def dtshift(kernel, ndim, sym=False):
    """Shift a potentially symmetric kernel

    Parameters
    ----------
    kernel : ([D], D, *shape) tensor
        Input kernel.
        If it has a single batch dimension, it is a diagonal kernel.
        Otherwise it is a full kernel. If `sym`, diagonal element are
        assumed to have an odd symmetry (DCT-I) and off-diagonal elements
        are assumed to have an odd antisymmetry (DST-I).
    ndim : int
        Number of spatial dimensions.
    sym : bool
        Symmetric kernel.

    Returns
    -------
    kernel : ([D], D, *shape) tensor
        Shifted kernel

    """
    if not sym:
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
    slicer = tuple(slice(int(floor(s / 2)), None) for s in shape)
    if nbatch == 2:
        slicer0 = tuple(slice(int(floor(s / 2)), -1) for s in shape)
        slicer1 = tuple(slice(int(floor(s / 2)) + 1, None) for s in shape)
        for d in range(ndim):
            kernel[(d, d, *slicer)] = tmp[(d, d, *slicer)]
            for dd in range(ndim):
                if dd == d: continue
                kernel[(d, dd, *slicer0)] = tmp[(d, dd, *slicer1)]
    else:
        slicer = (slice(None),) * nbatch + slicer
        kernel[slicer] = tmp[slicer]
    kernel = torch.fft.ifftshift(kernel, list(range(-ndim, 0)))
    return kernel


def real(x):
    """Return the real part (even if already real)"""
    if x.is_complex():
        x = x.real
    return x