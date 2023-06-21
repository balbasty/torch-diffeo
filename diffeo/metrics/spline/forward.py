import torch
import itertools
from typing import Union
from diffeo.utils import make_vector, ensure_list
from diffeo.diffdiv import diff, div, diff1d, div1d
from diffeo.bounds import bound2dft, sliding2dft, has_sliding
from diffeo.padding import pad
from . import kernels


def absolute(grid, voxel_size=1, bound='circulant', order=3):
    """Precision matrix for the Absolute energy of a deformation grid

    Parameters
    ----------
    grid : (..., *spatial, dim) tensor
    voxel_size : [list of] float, default=1
    bound : [list of] str, default='circulant'
    order : int, default=3

    Returns
    -------
    field : (..., *spatial, dim) tensor

    """
    ndim = grid.shape[-1]
    kernel = kernels.absolute(ndim, voxel_size, order)
    return _conv(grid, kernel, bound)


def membrane(grid, voxel_size=1, bound='circulant', order=3):
    """Precision matrix for the Membrane energy of a deformation grid

    Parameters
    ----------
    grid : (..., *spatial, dim) tensor
    voxel_size : [list of] float, default=1
    bound : [list of] str, default='circulant'
    order : int, default=3

    Returns
    -------
    field : (..., *spatial, dim) tensor

    """
    ndim = grid.shape[-1]
    kernel = kernels.membrane(ndim, voxel_size, order)
    return _conv(grid, kernel, bound)


def bending(grid, voxel_size=1, bound='circulant', order=3):
    """Precision matrix for the Bending energy of a deformation grid

    Parameters
    ----------
    grid : (..., *spatial, dim) tensor
    voxel_size : [list of] float, default=1
    bound : str, default='circulant'
    order : int, default=3

    Returns
    -------
    field : (..., *spatial, dim) tensor

    """
    ndim = grid.shape[-1]
    kernel = kernels.bending(ndim, voxel_size, order)
    return _conv(grid, kernel, bound)


def lame_shear(grid, voxel_size=1, bound='circulant', order=3):
    """Precision matrix for the Shear component of the Linear-Elastic energy.

    Parameters
    ----------
    grid : (..., *spatial, dim) tensor
    voxel_size : [list of] float, default=1
    bound : str, default='circulant'
    order : int, default=3

    Returns
    -------
    grid : (..., *spatial, dim) tensor

    """
    ndim = grid.shape[-1]
    kernel = kernels.lame_shear(ndim, voxel_size, order)
    return _conv(grid, kernel, bound)


def lame_div(grid, voxel_size=1, bound='circulant', order=3):
    """Precision matrix for the Divergence component of the Linear-Elastic energy.

    Parameters
    ----------
    grid : (..., *spatial, dim) tensor
    voxel_size : [list of] float, default=1
    bound : [list of] str, default='circulant'
    order : int, default=3

    Returns
    -------
    grid : (..., *spatial, dim) tensor

    """
    ndim = grid.shape[-1]
    kernel = kernels.lame_div(ndim, voxel_size, order)
    return _conv(grid, kernel, bound)


def mixture(v, absolute=0, membrane=0, bending=0, lame_shear=0, lame_div=0,
            factor=1, voxel_size=1, bound='circulant', order=3):
    """Precision matrix for a mixture of energies for a deformation grid.

    Parameters
    ----------
    v : (..., *spatial, dim) tensor
    absolute : float, default=0
    membrane : float, default=0
    bending : float, default=0
    lame_shear : float, default=0
    lame_div : float, default=0
    factor : float, default=1
    voxel_size : [list of] float, default=1
    bound : [list of] str, default='circulant'
    order : int, default=3

    Returns
    -------
    Lv : (..., *spatial, dim) tensor

    """
    ndim = v.shape[-1]
    kernel = kernels.mixture(
        ndim, absolute, membrane, bending, lame_shear, lame_div, factor,
        voxel_size=voxel_size, order=order, dtype=v.dtype, device=v.device,
    )
    return _conv(v, kernel, bound)


def _conv(x, kernel, bound='circular'):
    """
    Parameters
    ----------
    x : ([B], *shape, ndim) tensor
    kernel : ([ndim], ndim, *kernel_size) tensor
    bound : str

    Returns
    -------
    y : ([B], *shape, ndim) tensor
    """
    ndim = x.shape[-1]
    kernel_size = kernel.shape[-1]
    conv = getattr(functional, f'conv{ndim}d')

    add_batch = x.ndim == ndim + 1
    if add_batch:
        x = x[None]
    x = x.movedim(-1, 1)

    if has_sliding(bound):
        y = _pad_sliding(x, [(kernel_size - 1) // 2] * ndim, mode=bound)
    else:
        y = pad(x, [(kernel_size - 1) // 2] * ndim, mode=bound, side='both')

    kernel = kernel.to(x)
    if kernel.ndim == ndim + 2:
        # linear elastic -> (3, 3, 7, 7, 7) kernel
        y = conv(y, kernel)
        y = y[0].movedim(0, -1)
    else:
        # absolute/membrane/bending -> (3, 7, 7, 7) kernel
        assert kernel.ndim == ndim + 1
        y = conv(y, kernel[:, None], groups=ndim)

    y = y.movedim(1, -1)
    if add_batch:
        y = y.squeeze(0)
    return y


def _pad_sliding(x, padding, mode):
    ndim = len(padding)
    y = []
    for d, x1 in enumerate(x.unbind(-ndim-1)):
        y.append(pad(x1, padding, mode=sliding2dft(mode, d), side='both'))
    y = torch.stack(y, dim=-ndim-1)
    return y
