from interpol import (
    grid_push as _push,
    grid_count as _count,
    add_identity_grid as add_identity
)
from diffeo.utils import ensure_list
from diffeo.bounds import bound2dft
import torch


def push(image, flow, shape=None, bound='dct2', has_identity=False):
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
    if not has_identity:
        flow = add_identity(flow)
    ndim = flow.shape[-1]
    if bound == 'sliding':
        assert image.shape[-1] == ndim
        image, image0 = [], image
        for d in range(ndim):
            bound = ['dst2' if dd == d else 'dct2' for dd in range(ndim)]
            image.append(_push(
                image0[..., d:d+1].movedim(-1, -ndim-1), flow, shape,
                bound=bound, interpolation=1, extrapolate=True))
        image = torch.cat(image, dim=ndim-1).movedim(-ndim-1, -1)
    else:
        bound = ensure_list(bound, ndim)
        bound = list(map(lambda x: bound2dft.get(x, x), bound))
        image = image.movedim(-1, -ndim-1)
        image = _push(image, flow, shape, bound=bound, interpolation=1, extrapolate=True)
        image = image.movedim(-ndim-1, -1)
    return image


def count(flow, shape=None, bound='dct2', has_identity=False):
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
    if not has_identity:
        flow = add_identity(flow)
    ndim = flow.shape[-1]
    if bound == 'sliding':
        image = []
        for d in range(ndim):
            bound = ['dst2' if dd == d else 'dct2' for dd in range(ndim)]
            image.append(_count(
                flow, shape,
                bound=bound, interpolation=1, extrapolate=True))
        image = torch.stack(image, dim=-1)
    else:
        bound = ensure_list(bound, ndim)
        bound = list(map(lambda x: bound2dft.get(x, x), bound))
        image = _count(flow, shape, bound=bound, interpolation=1, extrapolate=True)
        image = image.unsqueeze(-1)
    return image
