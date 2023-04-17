from jitfields.pushpull import pull as _pull
from diffeo.flows import add_identity
from diffeo.utils import ensure_list
from diffeo.bounds import bound2dft
import torch


def pull(image, flow, bound='dct2', has_identity=False):
    """Warp an image according to a (voxel) displacement field.

    Parameters
    ----------
    image : (..., *shape_in, C) tensor
        Input image.
    flow : (..., *shape_out, D) tensor
        Displacement field, in voxels.
    bound : {'dft', 'dct[1|2|3|4]', 'dft[1|2|3|4]'}, default='dct2'
        Boundary conditions.
    has_identity : bool, default=False
        - If False, `flow` is contains relative displacement.
        - If True, `flow` contains absolute coordinates.

    Returns
    -------
    warped : (..., *shape_out, C) tensor
        Warped image

    """
    if not has_identity:
        flow = add_identity(flow)
    ndim = flow.shape[-1]
    if bound == 'sliding':
        assert image.shape[-1] == ndim
        image, image0 = [], image
        for d in range(ndim):
            bound = ['dst2' if dd == d else 'dct2' for dd in range(ndim)]
            image.append(_pull(
                image0[..., d:d+1], flow,
                bound=bound, order=1, extrapolate=True))
        return torch.cat(image, dim=-1)
    else:
        bound = ensure_list(bound, ndim)
        bound = list(map(lambda x: bound2dft.get(x, x), bound))
        return _pull(image, flow, bound=bound, order=1, extrapolate=True)
