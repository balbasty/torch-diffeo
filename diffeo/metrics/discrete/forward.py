import torch
import itertools
from typing import Union
from diffeo.utils import make_vector, ensure_list
from diffeo.diffdiv import diff, div, diff1d, div1d


def absolute(grid, voxel_size=1, bound='dft'):
    """Precision matrix for the Absolute energy of a deformation grid

    Parameters
    ----------
    grid : (..., *spatial, dim) tensor
    voxel_size : float or sequence[float], default=1

    Returns
    -------
    field : (..., *spatial, dim) tensor

    """
    backend = dict(dtype=grid.dtype, device=grid.device)
    dim = grid.shape[-1]
    voxel_size = make_vector(voxel_size, dim, **backend)
    grid = grid * voxel_size.square()
    return grid


def membrane(grid, voxel_size=1, bound='dft'):
    """Precision matrix for the Membrane energy of a deformation grid

    Parameters
    ----------
    grid : (..., *spatial, dim) tensor
    voxel_size : float or sequence[float], default=1
    bound : str, default='dft'

    Returns
    -------
    field : (..., *spatial, dim) tensor

    """
    backend = dict(dtype=grid.dtype, device=grid.device)
    dim = grid.shape[-1]
    voxel_size = make_vector(voxel_size, dim, **backend)
    grid = grid.movedim(-1, -(dim + 1))

    dims = list(range(grid.dim()-dim, grid.dim()))
    grid = diff(grid, dim=dims, voxel_size=voxel_size, side='f', bound=bound)
    dims = list(range(grid.dim()-1-dim, grid.dim()-1))
    grid = div(grid, dim=dims, voxel_size=voxel_size, side='f', bound=bound)

    grid = grid.movedim(-(dim + 1), -1)
    if (voxel_size != 1).any():
        grid.mul_(voxel_size.square())
    return grid


def bending(grid, voxel_size=1, bound='dft'):
    """Precision matrix for the Bending energy of a deformation grid

    Parameters
    ----------
    grid : (..., *spatial, dim) tensor
    voxel_size : float or sequence[float], default=1
    bound : str, default='dft'
    weights : (..., *spatial) tensor, optional

    Returns
    -------
    field : (..., *spatial, dim) tensor

    """
    backend = dict(dtype=grid.dtype, device=grid.device)
    ndim = grid.shape[-1]
    bound = ensure_list(bound, ndim)
    voxel_size = make_vector(voxel_size, ndim, **backend)
    grid = grid.movedim(-1, -(ndim + 1))

    dims = list(range(grid.dim()-ndim, grid.dim()))

    # allocate buffers
    if not grid.requires_grad:
        bufi = torch.empty_like(grid)
        bufij = torch.empty_like(grid)
        bufjj = torch.empty_like(grid)
    else:
        bufi = bufij = bufjj = None

    mom = 0
    for i in range(ndim):
        for side_i in ('f', 'b'):
            opti = dict(dim=dims[i], bound=bound[i], side=side_i,
                        voxel_size=voxel_size[i])
            di = diff1d(grid, **opti, out=bufi)
            for j in range(i, ndim):
                for side_j in ('f', 'b'):
                    optj = dict(dim=dims[j], bound=bound[j], side=side_j,
                                voxel_size=voxel_size[j])
                    dj = diff1d(di, **optj, out=bufij)
                    dj = div1d(dj, **optj, out=bufjj)
                    dj = div1d(dj, **opti, out=bufij)
                    if i != j:
                        # off diagonal -> x2  (upper + lower element)
                        dj = dj.mul_(2)
                    mom += dj
    grid = mom.div_(4.)

    grid = grid.movedim(-(ndim + 1), -1)
    if (voxel_size != 1).any():
        grid.mul_(voxel_size.square())
    return grid


def lame_shear(grid, voxel_size=1, bound='dft'):
    """Precision matrix for the Shear component of the Linear-Elastic energy.

    Parameters
    ----------
    grid : (..., *spatial, dim) tensor
    voxel_size : float or sequence[float], default=1
    bound : str, default='dft'

    Returns
    -------
    grid : (..., *spatial, dim) tensor

    """
    backend = dict(dtype=grid.dtype, device=grid.device)
    ndim = grid.shape[-1]
    bound = ensure_list(bound, ndim)
    voxel_size = make_vector(voxel_size, ndim, **backend)
    bound = ensure_list(bound, ndim)
    dims = list(range(grid.dim() - 1 - ndim, grid.dim() - 1))

    # allocate buffers
    if not grid.requires_grad:
        buf1 = torch.empty_like(grid[..., 0])
        buf2 = torch.empty_like(buf1)
        buf3 = torch.empty_like(buf1)
    else:
        buf1 = buf2 = buf3 = None

    mom = torch.zeros_like(grid)
    for i in range(ndim):
        # symmetric part
        x_i = grid[..., i]
        for j in range(i, ndim):
            for side_i in ('f', 'b'):
                opt_ij = dict(dim=dims[j], side=side_i, bound=bound[j],
                              voxel_size=voxel_size[j])
                diff_ij = diff1d(x_i, **opt_ij, out=buf1).mul_(voxel_size[i])
                if i == j:
                    # diagonal elements
                    mom[..., i].add_(div1d(diff_ij, **opt_ij, out=buf2),
                                     alpha=0.5)
                else:
                    # off diagonal elements
                    x_j = grid[..., j]
                    for side_j in ('f', 'b'):
                        opt_ji = dict(dim=dims[i], side=side_j, bound=bound[i],
                                      voxel_size=voxel_size[i])
                        diff_ji = diff1d(x_j, **opt_ji, out=buf2).mul_(
                            voxel_size[j])
                        diff_ji.add_(diff_ij, alpha=0.5)
                        mom[..., j].add_(div1d(diff_ji, **opt_ji, out=buf3),
                                         alpha=0.25)
                        mom[..., i].add_(div1d(diff_ji, **opt_ij, out=buf3),
                                         alpha=0.25)
                    del x_j
        del x_i
    del grid

    mom.mul_(2 * voxel_size)  # JA added an additional factor 2 to the kernel
    return mom


def lame_div(grid, voxel_size=1, bound='dft'):
    """Precision matrix for the Divergence component of the Linear-Elastic energy.

    Parameters
    ----------
    grid : (..., *spatial, dim) tensor
    voxel_size : float or sequence[float], default=1 (actually unused)
    bound : str, default='dft'

    Returns
    -------
    grid : (..., *spatial, dim) tensor

    """
    ndim = grid.shape[-1]
    bound = ensure_list(bound, ndim)
    dims = list(range(grid.dim() - 1 - ndim, grid.dim() - 1))

    # precompute gradients
    grad = [dict(f={}, b={}) for _ in range(ndim)]
    opt = [dict(f={}, b={}) for _ in range(ndim)]
    for i in range(ndim):
        x_i = grid[..., i]
        for side in ('f', 'b'):
            opt_i = dict(dim=dims[i], side=side, bound=bound[i])
            grad[i][side] = diff1d(x_i, **opt_i)
            opt[i][side] = opt_i

    if not grid.requires_grad:
        buf1 = torch.empty_like(grid[..., 0])
        buf2 = torch.empty_like(grid[..., 0])
    else:
        buf1 = buf2 = None

    # compute divergence
    mom = torch.zeros_like(grid)
    all_sides = list(itertools.product(['f', 'b'], repeat=ndim))
    for sides in all_sides:
        div = buf1.zero_() if buf1 is not None else 0
        for i, side in enumerate(sides):
            div += grad[i][side]
        for i, side in enumerate(sides):
            mom[..., i] += div1d(div, **(opt[i][side]), out=buf2)

    mom /= float(2 ** ndim)  # weight sides combinations
    return mom


_absolute = absolute
_membrane = membrane
_bending = bending
_lame_shear = lame_shear
_lame_div = lame_div


def mixture(v, absolute=0, membrane=0, bending=0, lame_shear=0, lame_div=0,
            factor=1, voxel_size=1, bound='dft'):
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
    voxel_size : [sequence of] float, default=1
    bound : str, default='dft'

    Returns
    -------
    Lv : (..., *spatial, dim) tensor

    """
    backend = dict(dtype=v.dtype, device=v.device)
    dim = v.shape[-1]

    a0 = not is_zero(absolute)
    m0 = not is_zero(membrane)
    b0 = not is_zero(bending)
    s0 = not is_zero(lame_shear)
    d0 = not is_zero(lame_div)

    voxel_size = make_vector(voxel_size, dim, **backend)
    absolute = absolute * factor
    membrane = membrane * factor
    bending = bending * factor
    lame_shear = lame_shear * factor
    lame_div = lame_div * factor
    fdopt = dict(bound=bound, voxel_size=voxel_size)

    y = torch.zeros_like(v)
    if a0:
        y.add_(_absolute(v, **fdopt), alpha=absolute)
    if m0:
        y.add_(_membrane(v, **fdopt), alpha=membrane)
    if b0:
        y.add_(_bending(v, **fdopt), alpha=bending)
    if s0:
        y.add_(_lame_shear(v, **fdopt), alpha=lame_shear)
    if d0:
        y.add_(_lame_div(v, **fdopt), alpha=lame_div)

    return y


def is_zero(x: Union[float, torch.Tensor]) -> bool:
    if torch.is_tensor(x) and x.requires_grad:
        return False
    else:
        return x == 0
