from jitfields.pushpull import push as _push, count as _count
from diffeo.flows import add_identity


def push(image, flow, shape=None, bound='dct2', has_identity=False):
    """Splat an image according to a (voxel) displacement field.

    Parameters
    ----------
    image : (B, *shape_in, C) tensor
        Input image.
    flow : ([B], *shape_out, D) tensor
        Displacement field, in voxels.
    shape : list[int], optional
        Output shape
    bound : {'dft', 'dct[1|2|3|4]', 'dft[1|2|3|4]'}, default='dct2'
        Boundary conditions.
    has_identity : bool, default=False
        - If False, `flow` is contains relative displacement.
        - If True, `flow` contains absolute coordinates.

    Returns
    -------
    pushed : (B, *shape_out, C) tensor
        Pushed image

    """
    if not has_identity:
        flow = add_identity(flow)
    return _push(image, flow, shape, bound=bound, order=1, extrapolate=True)


def count(flow, shape=None, bound='dct2', has_identity=False):
    """Splat an image of ones according to a (voxel) displacement field.

    Parameters
    ----------
    flow : ([B], *shape_out, D) tensor
        Displacement field, in voxels.
    shape : list[int], optional
        Output shape
    bound : {'dft', 'dct[1|2|3|4]', 'dft[1|2|3|4]'}, default='dct2'
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
    return _count(flow, shape, bound=bound, order=1, extrapolate=True).unsqueeze(-1)