from jitfields.pushpull import push as _push, count as _count
from diffeo.flows import add_identity
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
    bound : {'dft', 'dct[1|2|3|4]', 'dst[1|2|3|4]'}, default='dct2'
        Boundary conditions.
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
                image0[..., d:d+1], flow, shape,
                bound=bound, order=1, extrapolate=True))
        return torch.cat(image, dim=-1)
    else:
        bound = ensure_list(bound, ndim)
        bound = list(map(lambda x: bound2dft.get(x, x), bound))
        return _push(image, flow, shape, bound=bound, order=1, extrapolate=True)


def count(flow, shape=None, bound='dct2', has_identity=False):
    """Splat an image of ones according to a (voxel) displacement field.

    Parameters
    ----------
    flow : ([B], *shape_out, D) tensor
        Displacement field, in voxels.
    shape : list[int], optional
        Output shape
    bound : {'dft', 'dct[1|2|3|4]', 'dst[1|2|3|4]'}, default='dct2'
        Boundary conditions.
    has_identity : bool, default=False
        - If False, `flow` is contains relative displacement.
        - If True, `flow` contains absolute coordinates.

    Returns
    -------
    pushed : (B, *shape_out, 1) tensor
        Pushed image

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
                bound=bound, order=1, extrapolate=True))
        return torch.stack(image, dim=-1)
    else:
        bound = ensure_list(bound, ndim)
        bound = list(map(lambda x: bound2dft.get(x, x), bound))
        return _count(flow, shape, bound=bound, order=1, extrapolate=True).unsqueeze(-1)
