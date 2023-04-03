from interpol import (
    grid_push as _push,
    grid_count as _count,
    add_identity_grid as add_identity
)


def push(image, flow, shape=None, bound='dct2', has_identity=False):
    """Splat an image according to a (voxel) displacement field.

    Parameters
    ----------
    image : ([B], *shape_in, C) tensor
        Input image.
        If input dtype is integer, assumes labels: each unique labels
        gets warped using linear interpolation, and the label map gets
        reconstructed by argmax.
    flow : ([B], *shape_out, D) tensor
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
    pushed : ([B], *shape_out, C) tensor
        Pushed image

    """
    if not has_identity:
        flow = add_identity(flow)
    ndim = flow.shape[-1]
    image = image.movedim(-1, -ndim-1)
    image = _push(image, flow, shape, bound=bound, interpolation=1, extrapolate=True)
    return image.movedim(-ndim-1, -1)


def count(flow, shape=None, bound='dct2', has_identity=False):
    """Splat an image of ones according to a (voxel) displacement field.

    Parameters
    ----------
    flow : ([B], *shape_out, D) tensor
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
    pushed : (B, *shape_out, 1) tensor
        Pushed image

    """
    if not has_identity:
        flow = add_identity(flow)
    return _count(flow, shape, bound=bound, interpolation=1, extrapolate=True).unsqueeze(-1)
