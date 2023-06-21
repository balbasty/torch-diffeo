from jitfields import resize as _resize
from diffeo.utils import ensure_list
from diffeo.bounds import bound2dft, has_sliding, sliding2dft
import torch


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
    if shape and factor:
        raise ValueError('Only one of `shape` and `factor` should be used.')

    bound = ensure_list(bound, ndim)
    bound = list(map(lambda x: bound2dft.get(x, x), bound))

    if has_sliding(bound):
        assert image.shape[-1] == ndim
        image0, image = image, []
        for d, image1 in enumerate(image0.unbind(-1)):
            image.append(
                _resize(
                    image1,
                    factor=factor,
                    shape=shape,
                    ndim=ndim,
                    bound=sliding2dft(bound, d),
                    anchor=anchor,
                    order=order,
                    prefilter=False,
                )
            )
        image = torch.stack(image, dim=-1)
    else:
        image = image.movedim(-1, -ndim-1)
        image = _resize(
            image,
            factor=factor,
            shape=shape,
            ndim=ndim,
            bound=bound,
            anchor=anchor,
            order=order,
            prefilter=False,
        )
        image = image.movedim(-ndim-1, -1)

    return image
