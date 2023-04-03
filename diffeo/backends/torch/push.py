from diffeo.flows import jacobian, jacdet, add_identity_, sub_identity
from .pull import pull


def push(image, flow, shape=None, bound='dct2', has_identity=False):
    """Splat an image according to a (voxel) displacement field.

    /!\ The torch version of `push` uses a small deformation approximation
    /!\ It also does not support the `shape` keyword`

    Parameters
    ----------
    image : (..., *shape_in, C) tensor
        Input image.
    flow : (..., *shape_out, D) tensor
        Displacement field, in voxels.
        Note that the order of the last dimension is inverse of what's
        usually expected in torch's grid_sample.
    shape : list[int], optional
        Output shape
    bound : {'dft', 'dct[1|2|3|4]', 'dft[1|2|3|4]'}, default='dct2'
        Boundary conditions. PyTorch does not really support them,
        so "reflection" (which is equivalent to "dct2") is always used.
    has_identity : bool, default=False
        - If False, `flow` is contains relative displacement.
        - If True, `flow` contains absolute coordinates.

    Returns
    -------
    pushed : (..., *shape_out, C) tensor
        Pushed image

    """
    if not has_identity:
        flow = flow.neg()
    else:
        flow = sub_identity(flow).neg_()
    dt = jacdet(flow, bound=bound, has_identity=False, add_identity=True)
    flow = add_identity_(flow)
    image = pull(image, flow, bound=bound, has_identity=True)
    image = image * dt.unsqueeze(-1)
    return image


def count(flow, shape=None, bound='dct2', has_identity=False):
    """Splat an image of ones according to a (voxel) displacement field.

    /!\ The torch version of `push` uses a small deformation approximation
    /!\ It also does not support the `shape` keyword`

    Parameters
    ----------
    flow : ([B], *shape_out, D) tensor
        Displacement field, in voxels.
    shape : list[int], optional
        Output shape
    bound : {'dft', 'dct[1|2|3|4]', 'dft[1|2|3|4]'}, default='dct2'
        Boundary conditions. PyTorch does not really support them,
        so "reflection" (which is equivalent to "dct2") is always used.
    has_identity : bool, default=False
        - If False, `flow` is contains relative displacement.
        - If True, `flow` contains absolute coordinates.

    Returns
    -------
    pushed : (B, *shape_out, 1) tensor
        Pushed image

    """
    if not has_identity:
        flow = flow.neg()
    else:
        flow = sub_identity(flow).neg_()
    dt = jacdet(flow, bound=bound, has_identity=False, add_identity=True)
    return dt.unsqueeze(-1)
