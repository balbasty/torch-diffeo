from interpol import grid_pull as _pull, add_identity_grid as add_identity
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
    bound : {'dft', 'dct{1|2|3|4}', 'dst{1|2|3|4}'}, default='dct2'
        Boundary conditions.
        Can also be one for {'circulant', 'neumann', 'dirichlet', 'sliding'},
        in which case the image is assumed to be a flow field.
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
                image0[..., d:d+1].movedim(-1, -ndim-1), flow,
                bound=sliding2dft(bound, d), interpolation=order, extrapolate=True))
        image = torch.cat(image, dim=ndim-1).movedim(-ndim-1, -1)
    else:
        image = image.movedim(-1, -ndim-1)
        image = _pull(image, flow, bound=bound, interpolation=order, extrapolate=True)
        image = image.movedim(-ndim-1, -1)
    return image
