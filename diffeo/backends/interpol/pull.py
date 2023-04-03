from interpol import grid_pull as _pull, add_identity_grid as add_identity


def pull(image, flow, bound='dct2', has_identity=False):
    """Warp an image according to a (voxel) displacement field.

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
    bound : {'dft', 'dct[1|2|3|4]', 'dft[1|2|3|4]'}, default='dct2'
        Boundary conditions. PyTorch does not really support them,
        so "reflection" (which is equivalent to "dct2") is always used.
    has_identity : bool, default=False
        - If False, `flow` is contains relative displacement.
        - If True, `flow` contains absolute coordinates.

    Returns
    -------
    warped : ([B], *shape_out, C) tensor
        Warped image

    """
    if not has_identity:
        flow = add_identity(flow)
    ndim = flow.shape[-1]
    image = image.movedim(-1, -ndim-1)
    image = _pull(image, flow, bound=bound, interpolation=1, extrapolate=True)
    return image.movedim(-ndim-1, -1)
