from jitfields import spline_coeff_nd
from diffeo.utils import ensure_list
from diffeo.flows import identity
from diffeo.bounds import bound2dft, has_sliding, sliding2dft
from .pull import pull
import torch


def to_coeff_(ndim, image, bound='dct2', order=1):
    """Compute interpolating spline coefficients, in place.

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions
    image : (..., *shape, C) tensor
        Input image
    bound : {'dft', 'dct{1|2|3|4}', 'dst{1|2|3|4}'}, default='dct2'
        Boundary conditions.
        Can also be one for {'circulant', 'neumann', 'dirichlet', 'sliding'},
        in which case the image is assumed to be a flow field.
    order : int, default=1
        Order of the spline encoding the input `image`.
        Should generally only be used if `image` is a flow field.

    Returns
    -------
    coeff : (..., *shape, C) tensor
        Spline coefficients

    """
    if order in (0, 1):
        return image
    bound = ensure_list(bound, ndim)
    bound = list(map(lambda x: bound2dft.get(x, x), bound))
    if has_sliding(bound):
        assert image.shape[-1] == ndim
        for d, image1 in enumerate(image.unbind(-1)):
            image1.copy_(spline_coeff_nd(
                image1,
                bound=sliding2dft(bound, d), order=order, ndim=ndim))
    else:
        image = image.movedim(-1, -ndim-1)
        image.copy_(spline_coeff_nd(
            image, bound=bound, order=order, ndim=ndim))
        image = image.movedim(-ndim-1, -1)
    return image


def to_coeff(ndim, image, bound='dct2', order=1):
    """Compute interpolating spline coefficients.

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions
    image : (..., *shape, C) tensor
        Input image
    bound : {'dft', 'dct{1|2|3|4}', 'dst{1|2|3|4}'}, default='dct2'
        Boundary conditions.
        Can also be one for {'circulant', 'neumann', 'dirichlet', 'sliding'},
        in which case the image is assumed to be a flow field.
    order : int, default=1
        Order of the spline encoding the input `image`.
        Should generally only be used if `image` is a flow field.

    Returns
    -------
    coeff : (..., *shape, C) tensor
        Spline coefficients

    """
    if order in (0, 1):
        return image
    return to_coeff_(ndim, image.clone(), bound, order)


def from_coeff(ndim, coeff, bound='dct2', order=1):
    """Evaluate spline coefficients at integer nodes

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions
    coeff : (..., *shape, C) tensor
        Input spline coefficients
    bound : {'dft', 'dct{1|2|3|4}', 'dst{1|2|3|4}'}, default='dct2'
        Boundary conditions.
        Can also be one for {'circulant', 'neumann', 'dirichlet', 'sliding'},
        in which case the image is assumed to be a flow field.
    order : int, default=1
        Order of the spline encoding the input `image`.
        Should generally only be used if `image` is a flow field.

    Returns
    -------
    image : (..., *shape, C) tensor
        Interpolated image

    """
    if order in (0, 1):
        return image
    id = identity(coeff.shape[-ndim-1:-1], dtype=coeff.dtype, device=coeff.device)
    return pull(coeff, id, has_identity=True, order=order, bound=bound)
