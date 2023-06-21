from jitfields.pushpull import pull as _pull
from diffeo.flows import add_identity
from diffeo.utils import ensure_list
from diffeo.bounds import bound2dft, has_sliding, sliding2dft
import torch


def pull(image, flow, bound='dct2', has_identity=False, order=1):
    """Warp an image according to a (voxel) displacement field.

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
    order : int, default=1
        Order of the spline encoding the input `image`.
        Should generally only be used if `image` is a flow field.

    Returns
    -------
    warped : (..., *shape_out, C) tensor
        Warped image

    """
    if not has_identity:
        flow = add_identity(flow)
    ndim = flow.shape[-1]
    bound = ensure_list(bound, ndim)
    bound = list(map(lambda x: bound2dft.get(x, x), bound))
    if has_sliding(bound):
        assert image.shape[-1] == ndim
        image, image0 = [], image
        for d in range(ndim):
            image.append(_pull(
                image0[..., d:d+1], flow,
                bound=sliding2dft(bound, d), order=order, extrapolate=True))
        return torch.cat(image, dim=-1)
    else:
        return _pull(image, flow, bound=bound, order=order, extrapolate=True)
