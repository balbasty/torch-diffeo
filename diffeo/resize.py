import torch
from torch.nn import functional as F
from diffeo.backends import default_backend


def upsample(ndim, image, factor=None, shape=None, anchor='center',
             bound='circulant', order=1, backend=None):
    """Upsample using centers or edges of the corner voxels as anchors.

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions
    image : (..., *shape_in, C) tensor
        Input image
    factor OR shape : int or list[int]
        Either the upsampling factor (> 1) the output shape.
    anchor : {'center', 'edge'}
        Align either the centers or edges of the corner voxels across levels.
    bound : str
        Boundary conditions.
    order : int
        Order of the encoding splines.
    backend : diffeo.backend
        Which interpolation backend to use.

    Returns
    -------
    image : (..., *shape_out, C) tensor
        Upsampled image

    """
    backend = backend or default_backend
    return backend.resize(
        ndim=ndim,
        image=image,
        factor=factor,
        shape=shape,
        anchor=anchor,
        bound=bound,
        order=order,
    )


def downsample(ndim, image, factor=None, shape=None, anchor='center', bound='circulant', order=1, backend=None):
    """Downsample using centers or edges of the corner voxels as anchors.

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions
    image : (..., *shape_in, C) tensor
        Input image
    factor OR shape : int or list[int]
        Either the downsampling factor (> 1) the output shape.
    anchor : {'center', 'edge'}
        Align either the centers or edges of the corner voxels across levels.
    bound : str
        Boundary conditions.
    order : int
        Order of the encoding splines.
    backend : diffeo.backend
        Which interpolation backend to use.

    Returns
    -------
    image : (..., *shape_out, C) tensor
        Downsampled image

    """
    if factor:
        if isinstance(factor, (list, tuple)):
            factor = [1/f for f in factor]
        else:
            factor = 1/factor
    return upsample(ndim, image, factor, shape, anchor, bound, order, backend)


def upsample_flow(flow, factor=None, shape=None, anchor='center',
                  bound='circulant', order=1, backend=None):
    """Upsample a flow field  using centers or edges of the corner
    voxels as anchors.

    Parameters
    ----------
    flow : (..., *shape_in, D) tensor
        Input image
    factor OR shape : int or list[int]
        Either the upsampling factor (> 1) the output shape.
    anchor : {'center', 'edge'}
        Align either the centers or edges of the corner voxels across levels.
    bound : str
        Boundary conditions.
    order : int
        Order of the encoding splines.
    backend : diffeo.backend
        Which interpolation backend to use.

    Returns
    -------
    flow : (..., *shape_out, D) tensor
        Upsampled flow

    """
    *shape_in, ndim = flow.shape[1:-1]

    # upsample flow
    flow = upsample(ndim, flow, factor, shape, anchor, bound, order, backend)

    # compute scale
    shape_out = flow.shape[1:-1]
    if anchor[0].lower() == 'c':
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


def downsample_flow(flow, factor=None, shape=None, anchor='center',
                    bound='circulant', order=1, backend=None):
    """Downsample a flow field  using centers or edges of the corner
    voxels as anchors.

    Parameters
    ----------
    flow : (..., *shape_in, D) tensor
        Input image
    factor OR shape : int or list[int]
        Either the downsampling factor (> 1) the output shape.
    anchor : {'center', 'edge'}
        Align either the centers or edges of the corner voxels across levels.
    bound : str
        Boundary conditions.
    order : int
        Order of the encoding splines.
    backend : diffeo.backend
        Which interpolation backend to use.

    Returns
    -------
    flow : (..., *shape_out, D) tensor
        Downsampled flow

    """
    if factor:
        if isinstance(factor, (list, tuple)):
            factor = [1/f for f in factor]
        else:
            factor = 1/factor
    return upsample_flow(flow, factor, shape, anchor, bound, order, backend)
