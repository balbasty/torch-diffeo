__all__ = [
    # 1D
    'dct', 'dst', 'idct', 'idst',
    # ND
    'dctn', 'dstn', 'idctn', 'idstn',
    # convenience wrappers
    'dct1', 'dct2', 'dct3', 'dct4',
    'dst1', 'dst2', 'dst3', 'dst4',
    'idct1', 'idct2', 'idct3', 'idct4',
    'idst1', 'idst2', 'idst3', 'idst4',
]
from .realtransforms_autograd import DCTN, DSTN, flipnorm, fliptype
from torch import Tensor
from typing import Optional
_IMPLEMENTED_TYPES = (1, 2, 3)


def dct(
    x: Tensor,
    dim: int = -1,
    norm: str = 'backward',
    type: int = 2,
) -> Tensor:
    """Return the Discrete Cosine Transform

    !!! warning
        Type IV not implemented

    Parameters
    ----------
    x : tensor
        The input tensor
    dim : int
        Dimensions over which the DCT is computed.
        Default is the last one.
    norm : {“backward”, “ortho”, “forward”}
        Normalization mode. Default is “backward”.
    type: {1, 2, 3, 4}
        Type of the DCT. Default type is 2.

    Returns
    -------
    y : tensor
        The transformed tensor.

    """
    if dim is None:
        dim = -1
    if type in _IMPLEMENTED_TYPES:
        return DCTN.apply(x, type, dim, norm)
    else:
        raise ValueError('DCT only implemented for types I-IV')


def idct(
    x: Tensor,
    dim: int = -1,
    norm: str = 'backward',
    type: int = 2,
) -> Tensor:
    """Return the Inverse Discrete Cosine Transform

    !!! warning
        Type IV not implemented

    Parameters
    ----------
    x : tensor
        The input tensor
    dim : int
        Dimensions over which the DCT is computed.
        Default is the last one.
    norm : {“backward”, “ortho”, “forward”}
        Normalization mode. Default is “backward”.
    type: {1, 2, 3, 4}
        Type of the DCT. Default type is 2.

    Returns
    -------
    y : tensor
        The transformed tensor.

    """
    if dim is None:
        dim = -1
    norm = flipnorm[norm or "backward"]
    type = fliptype[type]
    return dct(x, dim, norm, type)


def dst(
    x: Tensor,
    dim: int = -1,
    norm: str = 'backward',
    type: int = 2,
) -> Tensor:
    """Return the Discrete Sine Transform

    !!! warning
        Type IV not implemented

    !!! warning
        `dst(..., norm="ortho")` yields a different result than `scipy`
        and `cupy` for types 2 and 3. This is because their DST is not
        properly orthogonalized. Use `norm="ortho_scipy"` to get results
        matching their implementation.

    Parameters
    ----------
    x : tensor
        The input tensor
    dim : int
        Dimensions over which the DCT is computed.
        Default is the last one.
    norm : {“backward”, “ortho”, “forward”, "ortho_scipy"}
        Normalization mode. Default is “backward”.
    type: {1, 2, 3, 4}
        Type of the DCT. Default type is 2.

    Returns
    -------
    y : tensor
        The transformed tensor.

    """
    if dim is None:
        dim = -1
    if type in _IMPLEMENTED_TYPES:
        return DSTN.apply(x, type, dim, norm)
    else:
        raise ValueError('DST only implemented for types I-IV')


def idst(
    x: Tensor,
    dim: int = -1,
    norm: str = 'backward',
    type: int = 2,
) -> Tensor:
    """Return the Inverse Discrete Sine Transform

    !!! warning
        Type IV not implemented

    !!! warning
        `idst(..., norm="ortho")` yields a different result than `scipy`
        and `cupy` for types 2 and 3. This is because their DST is not
        properly orthogonalized. Use `norm="ortho_scipy"` to get results
        matching their implementation.

    Parameters
    ----------
    x : tensor
        The input tensor
    dim : int
        Dimensions over which the DCT is computed.
        Default is the last one.
    norm : {“backward”, “ortho”, “forward”, "ortho_scipy"}
        Normalization mode. Default is “backward”.
    type: {1, 2, 3, 4}
        Type of the DCT. Default type is 2.

    Returns
    -------
    y : tensor
        The transformed tensor.

    """
    if dim is None:
        dim = -1
    norm = flipnorm[norm or "backward"]
    type = fliptype[type]
    return dst(x, dim, norm, type)


def dctn(
    x: Tensor,
    dim: Optional[int] = None,
    norm: str = 'backward',
    type: int = 2,
) -> Tensor:
    """Return multidimensional Discrete Cosine Transform
    along the specified axes.

    !!! warning
        Type IV not implemented

    Parameters
    ----------
    x : tensor
        The input tensor
    dim : [sequence of] int
        Dimensions over which the DCT is computed.
        If not given, all dimensions are used.
    norm : {“backward”, “ortho”, “forward”}
        Normalization mode. Default is “backward”.
    type: {1, 2, 3, 4}
        Type of the DCT. Default type is 2.

    Returns
    -------
    y : tensor
        The transformed tensor.

    """
    if dim is None:
        dim = list(range(x.ndim))
    if type in _IMPLEMENTED_TYPES:
        return DCTN.apply(x, type, dim, norm)
    else:
        raise ValueError('DCT only implemented for types I-IV')


def idctn(
    x: Tensor,
    dim: Optional[int] = None,
    norm: str = 'backward',
    type: int = 2,
) -> Tensor:
    """Return multidimensional Inverse Discrete Cosine Transform
    along the specified axes.

    !!! warning
        Type IV not implemented

    Parameters
    ----------
    x : tensor
        The input tensor
    dim : [sequence of] int
        Dimensions over which the DCT is computed.
        If not given, all dimensions are used.
    norm : {“backward”, “ortho”, “forward”}
        Normalization mode. Default is “backward”.
    type: {1, 2, 3, 4}
        Type of the DCT. Default type is 2.

    Returns
    -------
    y : tensor
        The transformed tensor.

    """
    if dim is None:
        dim = list(range(x.ndim))
    norm = flipnorm[norm or "backward"]
    type = fliptype[type]
    return dctn(x, dim, norm, type)


def dstn(
    x: Tensor,
    dim: Optional[int] = None,
    norm: str = 'backward',
    type: int = 2,
) -> Tensor:
    """Return multidimensional Discrete Sine Transform
    along the specified axes.

    !!! warning
        Type IV not implemented

    !!! warning
        `dstn(..., norm="ortho")` yields a different result than `scipy`
        and `cupy` for types 2 and 3. This is because their DST is not
        properly orthogonalized. Use `norm="ortho_scipy"` to get results
        matching their implementation.

    Parameters
    ----------
    x : tensor
        The input tensor
    dim : [sequence of] int
        Dimensions over which the DCT is computed.
        If not given, all dimensions are used.
    norm : {“backward”, “ortho”, “forward”, "ortho_scipy"}
        Normalization mode. Default is “backward”.
    type: {1, 2, 3, 4}
        Type of the DCT. Default type is 2.

    Returns
    -------
    y : tensor
        The transformed tensor.

    """
    if dim is None:
        dim = list(range(x.ndim))
    if type in _IMPLEMENTED_TYPES:
        return DSTN.apply(x, type, dim, norm)
    else:
        raise ValueError('DST only implemented for types I-IV')


def idstn(
    x: Tensor,
    dim: Optional[int] = None,
    norm: str = 'backward',
    type: int = 2,
) -> Tensor:
    """Return multidimensional Inverse Discrete Sine Transform
    along the specified axes.

    !!! warning
        Type IV not implemented

    !!! warning
        `idstn(..., norm="ortho")` yields a different result than `scipy`
        and `cupy` for types 2 and 3. This is because their DST is not
        properly orthogonalized. Use `norm="ortho_scipy"` to get results
        matching their implementation.

    Parameters
    ----------
    x : tensor
        The input tensor
    dim : [sequence of] int
        Dimensions over which the DCT is computed.
        If not given, all dimensions are used.
    norm : {“backward”, “ortho”, “forward”, "ortho_scipy}
        Normalization mode. Default is “backward”.
    type: {1, 2, 3, 4}
        Type of the DCT. Default type is 2.

    Returns
    -------
    y : tensor
        The transformed tensor.

    """
    if dim is None:
        dim = list(range(x.ndim))
    norm = flipnorm[norm or "backward"]
    type = fliptype[type]
    return dstn(x, dim, norm, type)


def dct1(x, dim:-1, norm=None):
    return dct(x, dim, norm, type=1)


def dct2(x, dim:-1, norm=None):
    return dct(x, dim, norm, type=2)


def dct3(x, dim:-1, norm=None):
    return dct(x, dim, norm, type=3)


def dct4(x, dim:-1, norm=None):
    return dct(x, dim, norm, type=4)


def idct1(x, dim:-1, norm=None):
    return idct(x, dim, norm, type=1)


def idct2(x, dim:-1, norm=None):
    return idct(x, dim, norm, type=2)


def idct3(x, dim:-1, norm=None):
    return idct(x, dim, norm, type=3)


def idct4(x, dim:-1, norm=None):
    return idct(x, dim, norm, type=4)


def dst1(x, dim:-1, norm=None):
    return dst(x, dim, norm, type=1)


def dst2(x, dim:-1, norm=None):
    return dst(x, dim, norm, type=2)


def dst3(x, dim:-1, norm=None):
    return dst(x, dim, norm, type=3)


def dst4(x, dim:-1, norm=None):
    return dst(x, dim, norm, type=4)


def idst1(x, dim:-1, norm=None):
    return idst(x, dim, norm, type=1)


def idst2(x, dim:-1, norm=None):
    return idst(x, dim, norm, type=2)


def idst3(x, dim:-1, norm=None):
    return idst(x, dim, norm, type=3)


def idst4(x, dim:-1, norm=None):
    return idst(x, dim, norm, type=4)
