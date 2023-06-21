from interpol import resize as _resize
from diffeo.utils import ensure_list
from diffeo.bounds import bound2dft, has_sliding, sliding2dft
import torch


def resize(ndim, image, factor=None, shape=None, anchor='center', bound='dct2', order=1):
    """Upsample using centers or edges of the corner voxels as anchors.

    Parameters
    ----------
    image : (B, *shape_in, C) tensor
    factor OR shape : int or list[int]
    anchor : {'center', 'edge'}

    Returns
    -------
    image : (B, *shape_out, C) tensor

    """
    if shape and factor:
        raise ValueError('Only one of `shape` and `factor` should be used.')

    batch = image.shape[:-ndim-1]
    *shape_inp, C = image.shape[-ndim-1]
    if len(batch) != 1:
        image = image.reshape([-1, *shape_inp, C])

    bound = ensure_list(bound, ndim)
    bound = list(map(lambda x: bound2dft.get(x, x), bound))

    if has_sliding(bound):
        assert image.shape[-1] == ndim
        image0, image = image, []
        for d, image1 in enumerate(image0.unbind(-1)):
            image.append(
                _resize(
                    image1.unsqueeze(1),
                    factor=factor,
                    shape=shape,
                    bound=sliding2dft(bound, d),
                    anchor=anchor,
                    interpolation=order,
                    prefilter=False,
                ).squeeze(1)
            )
        image = torch.stack(image, dim=-1)
    else:
        image = image.movedim(-1, -ndim-1)
        image = _resize(
            image,
            factor=factor,
            shape=shape,
            bound=bound,
            anchor=anchor,
            interpolation=order,
            prefilter=False,
        )
        image = image.movedim(-ndim-1, -1)

    if len(batch) != 1:
        shape_out = image.shape[-ndim-1:-1]
        image = image.reshape([*batch, *shape_out, C])
    return image
