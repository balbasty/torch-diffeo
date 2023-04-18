from jitfields import grad as _grad
from diffeo.flows import add_identity
from diffeo.utils import ensure_list
from diffeo.bounds import bound2dft
import torch


def grad(image, flow, bound='dct2', has_identity=False):
    """Compute spatial gradients of image according to a (voxel) displacement field.

    Parameters
    ----------
    image : (..., *shape_in, C) tensor
        Input image.
    flow : (..., *shape_out, D) tensor
        Displacement field, in voxels.
    bound : {'dft', 'dct[1|2|3|4]', 'dst[1|2|3|4]'}, default='dct2'
        Boundary conditions.
    has_identity : bool, default=False
        - If False, `flow` is contains relative displacement.
        - If True, `flow` contains absolute coordinates.

    Returns
    -------
    grad : (..., *shape_out, C, D) tensor
        Sampled gradients

    """
    if not has_identity:
        flow = add_identity(flow)
    ndim = flow.shape[-1]
    if bound == 'sliding':
        assert image.shape[-1] == ndim
        image, image0 = [], image
        for d in range(ndim):
            bound = ['dst2' if dd == d else 'dct2' for dd in range(ndim)]
            image.append(_grad(
                image0[..., d:d+1], flow,
                bound=bound, order=1, extrapolate=True))
        return torch.cat(image, dim=-2)
    else:
        bound = ensure_list(bound, ndim)
        bound = list(map(lambda x: bound2dft.get(x, x), bound))
        return _grad(image, flow, bound=bound, order=1, extrapolate=True)
