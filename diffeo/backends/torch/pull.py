import torch
from torch.nn import functional as F
from diffeo.utils import cartesian_grid, expand_shapes


def pull(image, flow, bound='dct2', has_identity=False, **kwargs):
    """Warp an image according to a (voxel) displacement field.

    Parameters
    ----------
    image : (..., *shape_in, C) tensor
        Input image.
    flow : (..., *shape_out, D) tensor
        Displacement field, in voxels.
        Note that the order of the last dimension is inverse of what's
        usually expected in torch's grid_sample.
    bound : {'dft', 'dct[1|2|3|4]', 'dst[1|2|3|4]'}, default='dct2'
        Boundary conditions. PyTorch does not really support them,
        so "reflection" (which is equivalent to "dct2") is always used.
    has_identity : bool, default=False
        - If False, `flow` is contains relative displacement.
        - If True, `flow` contains absolute coordinates.

    Returns
    -------
    warped : (..., *shape_out, C) tensor
        Warped image

    """
    kwargs.setdefault('align_corners', True)
    kwargs.setdefault('padding_mode', 'reflection')
    C, D = image.shape[-1], flow.shape[-1]
    shape_out = flow.shape[-D-1:-1]
    shape_inp = image.shape[-D-1:-1]
    batch = expand_shapes(image.shape[:-D-1], flow.shape[:-D-1])
    image = image.expand([*batch, *shape_inp, C])
    flow = flow.expand([*batch, *shape_out, D])
    if len(batch) != 1:
        image = image.reshape([-1, *shape_inp, C])
        flow = flow.reshape([-1, *shape_out, D])

    flow = flow_to_torch(flow, shape_inp,
                         align_corners=kwargs['align_corners'],
                         has_identity=has_identity)
    image = F.grid_sample(image.movedim(-1, 1), flow, **kwargs).movedim(1, -1)
    if len(batch) != 1:
        image = image.reshape([*batch, *shape_inp, C])
    return image


def flow_to_torch(flow, shape, align_corners=True, has_identity=False):
    """Convert a voxel displacement field to a torch sampling grid

    Parameters
    ----------
    flow : (..., *shape, D) tensor
        Displacement field
    shape : list[int] tensor
        Spatial shape of the input image
    align_corners : bool, default=True
        Torch's grid mode
    has_identity : bool, default=False
        If False, `flow` is contains relative displacement.
        If False, `flow` contains absolute coordinates.

    Returns
    -------
    grid : (..., *shape, D) tensor
        Sampling grid to be used with torch's `grid_sample`

    """
    backend = dict(dtype=flow.dtype, device=flow.device)
    # 1) reverse last dimension
    flow = torch.flip(flow, [-1])
    # 2) add identity grid
    if not has_identity:
        grid = cartesian_grid(shape, **backend)
        grid = list(reversed(grid))
        for d, g in enumerate(grid):
            flow[..., d].add_(g)
    shape = list(reversed(shape))
    # 3) convert coordinates
    for d, s in enumerate(shape):
        if align_corners:
            # (0, N-1) -> (-1, 1)
            flow[..., d].mul_(2/(s-1)).add_(-1)
        else:
            # (-0.5, N-0.5) -> (-1, 1)
            flow[..., d].mul_(2/s).add_(1/s-1)
    return flow

