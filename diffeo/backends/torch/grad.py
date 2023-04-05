from .pull import pull
from diffeo.flows import jacobian
from diffeo.diffdiv import diff


def grad(image, flow, bound='dct2', has_identity=False):
    """Compute spatial gradients of image according to a (voxel) displacement field.

    Parameters
    ----------
    image : (..., *shape_in, C) tensor
        Input image.
    flow : (..., *shape_out, D) tensor
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
    grad : (..., *shape_out, C, D) tensor
        Sampled gradients

    """
    ndim = flow.shape[-1]
    jac = jacobian(flow, bound=bound, has_identity=has_identity, add_identity=True)
    jac = jac.inverse()
    image = pull(image, flow, bound=bound, has_identity=has_identity)
    image = image.movedim(-1, -ndim-1)
    image = diff(image, dim=range(-ndim, 0), bound=bound)
    image = jac.unsqueeze(-ndim-2).matmul(image.unsqueeze(-1)).squeeze(-1)
    image = image.movedim(-ndim-2, -2)
    return image
