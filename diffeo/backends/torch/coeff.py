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
    raise NotImplementedError('backend "torch" does not implement higher order splines')


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
    raise NotImplementedError('backend "torch" does not implement higher order splines')


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
    raise NotImplementedError('backend "torch" does not implement higher order splines')
