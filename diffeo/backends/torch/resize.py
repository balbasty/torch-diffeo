import torch
from torch.nn import functional as F


def resize(ndim, image, factor=None, shape=None, anchor='center', bound='dct2', order=1):
    """Upsample using centers or edges of the corner voxels as anchors.

    Parameters
    ----------
    image : (..., *shape_in, C) tensor
    factor OR shape : int or list[int]
    anchor : {'center', 'edge'}

    Returns
    -------
    image : (..., *shape_out, C) tensor

    """
    if order > 1:
        raise NotImplementedError('backend "torch" does not implement higher order splines')
    if shape and factor:
        raise ValueError('Only one of `shape` and `factor` should be used.')
    mode = 'linear' if ndim == 1 else 'bilinear' if ndim == 2 else 'trilinear'
    align_corners = (anchor[0].lower() == 'c')
    recompute_scale_factor = factor is not None

    batch = image.shape[:-ndim - 1]
    *shape_inp, C = image.shape[-ndim - 1]
    if len(batch) != 1:
        image = image.reshape([-1, *shape_inp, C])

    image = image.movedim(-1, 1)
    image = F.interpolate(image, size=shape, scale_factor=factor,
                          mode=mode, align_corners=align_corners,
                          recompute_scale_factor=recompute_scale_factor)
    image = image.movedim(1, -1)

    if len(batch) != 1:
        shape_out = image.shape[-ndim - 1:-1]
        image = image.reshape([*batch, *shape_out, C])
    return image
