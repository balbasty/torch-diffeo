from jitfields import grad as _grad
from diffeo.flows import add_identity


def grad(image, flow, bound='dct2', has_identity=False):
    """Compute spatial gradients of image according to a (voxel) displacement field.

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
    grad : (..., *shape_out, C, D) tensor
        Sampled gradients

    """
    if not has_identity:
        flow = add_identity(flow)
    return _grad(image, flow, bound=bound, order=1, extrapolate=True)
