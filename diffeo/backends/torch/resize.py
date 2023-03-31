import torch
from torch.nn import functional as F


def downsample(image, factor=None, shape=None, anchor='center'):
    """Downsample using centers or edges of the corner voxels as anchors.

    Parameters
    ----------
    image : (B, C, *shape_in) tensor
    factor OR shape : int or list[int]
    anchor : {'center', 'edge'} tensor

    Returns
    -------
    image : (B, C, *shape_out)

    """
    if shape and factor:
        raise ValueError('Only one of `shape` and `factor` should be used.')
    ndim = image.dim() - 2
    mode = 'linear' if ndim == 1 else 'bilinear' if ndim == 2 else 'trilinear'
    align_corners = (anchor[0].lower() == 'c')
    recompute_scale_factor = factor is not None
    if factor:
        if isinstance(factor, (list, tuple)):
            factor = [1/f for f in factor]
        else:
            factor = 1/factor
    image = F.interpolate(image, size=shape, scale_factor=factor,
                          mode=mode, align_corners=align_corners,
                          recompute_scale_factor=recompute_scale_factor)
    return image


def upsample(image, factor=None, shape=None, anchor='center'):
    """Upsample using centers or edges of the corner voxels as anchors.

    Parameters
    ----------
    image : (B, C, *shape_in) tensor
    factor OR shape : int or list[int]
    anchor : {'center', 'edge'}

    Returns
    -------
    image : (B, C, *shape_out) tensor

    """
    if shape and factor:
        raise ValueError('Only one of `shape` and `factor` should be used.')
    ndim = image.dim() - 2
    mode = 'linear' if ndim == 1 else 'bilinear' if ndim == 2 else 'trilinear'
    align_corners = (anchor[0].lower() == 'c')
    recompute_scale_factor = factor is not None
    image = F.interpolate(image, size=shape, scale_factor=factor,
                          mode=mode, align_corners=align_corners,
                          recompute_scale_factor=recompute_scale_factor)
    return image


def downsample_flow(flow, factor=None, shape=None, anchor='center'):
    """Downsample a flow field  using centers or edges of the corner
    voxels as anchors.

    Parameters
    ----------
    flow : (B, *shape_in, D) tensor
    factor OR shape : int or list[int]
    anchor : {'center', 'edge'}

    Returns
    -------
    flow : (B, *shape_out, D) tensor

    """
    shape_in = flow.shape[1:-1]

    # downsample flow
    flow = flow.movedim(-1, 1)
    flow = downsample(flow, factor, shape, anchor)
    flow = flow.movedim(1, -1)

    # compute scale
    shape_out = flow.shape[1:-1]
    if anchor[0] == 'c':
        factor = [(fout - 1) / (fin - 1)
                  for fout, fin in zip(shape_out, shape_in)]
    else:
        factor = [fout / fin
                  for fout, fin in zip(shape_out, shape_in)]

    # rescale displacement
    ndim = flow.dim() - 2
    for d in range(ndim):
        flow[..., d] /= factor[d]

    return flow


def upsample_flow(flow, factor=None, shape=None, anchor='center'):
    """Upsample a flow field  using centers or edges of the corner
    voxels as anchors.

    Parameters
    ----------
    flow : (B, *shape_in, D) tensor
    factor OR shape : int or list[int]
    anchor : {'center', 'edge'}

    Returns
    -------
    flow : (B, *shape_out, D) tensor

    """
    shape_in = flow.shape[1:-1]

    # upsample flow
    flow = flow.movedim(-1, 1)
    flow = upsample(flow, factor, shape, anchor)
    flow = flow.movedim(1, -1)

    # compute scale
    shape_out = flow.shape[1:-1]
    if anchor[0] == 'c':
        factor = [(fout - 1) / (fin - 1)
                  for fout, fin in zip(shape_out, shape_in)]
    else:
        factor = [fout / fin
                  for fout, fin in zip(shape_out, shape_in)]

    # rescale displacement
    ndim = flow.dim() - 2
    for d in range(ndim):
        flow[..., d] /= factor[d]

    return flow