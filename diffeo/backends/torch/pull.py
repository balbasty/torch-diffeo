import torch
from torch.nn import functional as F
from diffeo.utils import cartesian_grid


def pull(image, flow, bound='dct2', has_identity=False, **kwargs):
    """Warp an image according to a (voxel) displacement field.

    Parameters
    ----------
    image : (B, *shape_in, C) tensor
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
    warped : (B, *shape_out, C) tensor
        Warped image

    """
    kwargs.setdefault('align_corners', True)
    kwargs.setdefault('padding_mode', 'reflection')
    image = image.movedim(1, 0)
    B, C, *shape_in = image.shape
    D = flow.shape[-1]
    if flow.dim() == D+1:
        flow = flow[None]
    shape_out = flow.shape[1:-1]
    flow = flow_to_torch(flow, shape_in,
                         align_corners=kwargs['align_corners'],
                         has_identity=has_identity)
    B = max(len(flow), len(image))
    if len(flow) != B:
        flow = flow.expand([B, *flow.shape[1:]])
    if len(image) != B:
        image = image.expand([B, *image.shape[1:]])
    if not image.dtype.is_floating_point:
        vmax = flow.new_full([B, C, *shape_out], -float('inf'))
        warped = image.new_zeros([B, C, *shape_out])
        for label in image.unique():
            w = F.grid_sample((image == label).to(flow), flow, **kwargs)
            warped[w > vmax] = label
            vmax = torch.maximum(vmax, w)
        return warped.movedim(-1, 1)
    else:
        return F.grid_sample(image, flow, **kwargs).movedim(-1, 1)


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

