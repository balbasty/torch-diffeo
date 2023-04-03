import torch
import math
from diffeo.utils import cartesian_grid, make_vector


def laplace(shape, voxel_size=1, dtype=None, device=None):
    """Compute the Greens function of the Laplace operator

    Parameters
    ----------
    shape : list[int]
    voxel_size : list[float], default=1
    dtype : torch.dtype, optional
    device : torch.device, optional

    Returns
    -------
    kernel : (*shape, ndim) tensor

    """

    ndim = len(shape)
    dtype = dtype or torch.get_default_dtype()
    voxel_size = make_vector(voxel_size, ndim, dtype=dtype, device=device)

    # Euclidean distance to center
    greens = torch.stack(cartesian_grid(shape, dtype=dtype, device=device), -1)
    center = (make_vector(shape, dtype=dtype, device=device) - 1) / 2
    greens = greens.sub_(center).mul_(voxel_size)
    greens = greens.square_().sum(-1).sqrt_()

    # make it well posed
    greens.masked_fill_(greens == 0, voxel_size.square().mean().sqrt().div(2))

    # Green's function
    if ndim == 2:
        greens = greens.log_().div_(2*math.pi)
    elif ndim == 3:
        greens = greens.reciprocal_().div_(-4*math.pi)
    else:
        raise ValueError('Laplace Greens kernel only implemented in 2D/3D')

    # Per component + voxel size scaling
    greens = torch.stack([greens]*3, -1)
    greens /= voxel_size.square()

    return greens


def helmoltz(shape, alpha=1e-3, voxel_size=1, dtype=None, device=None):
    """Compute the Greens function of the Helmoltz operator

    The Helmoltz operator can be seen as a regularised version of the
    Laplace operator

    Parameters
    ----------
    shape : list[int]
    alpha : float, default=1e-3
    voxel_size : list[float], default=1
    dtype : torch.dtype, optional
    device : torch.device, optional

    Returns
    -------
    kernel : (*shape, ndim) tensor

    """

    alpha = alpha ** 0.5
    ndim = len(shape)
    dtype = dtype or torch.get_default_dtype()
    voxel_size = make_vector(voxel_size, ndim, dtype=dtype, device=device)

    # Euclidean distance to center
    dist = torch.stack(cartesian_grid(shape, dtype=dtype, device=device), -1)
    center = (make_vector(shape, dtype=dtype, device=device) - 1) / 2
    dist = dist.sub_(center).mul_(voxel_size)
    dist = dist.square_().sum(-1).sqrt_()

    # make it well posed
    dist.masked_fill_(dist == 0, voxel_size.square().mean().sqrt().div(2))

    # Green's function
    if ndim == 2:
        # Hankel
        greens = H01(dist.mul_(alpha))

        # leading term
        greens = greens.div_(4j)

    elif ndim == 3:
        # Hankel
        greens = h02(dist.mul_(alpha))

        # leading term
        greens = greens.mul_(1j * alpha / (4*math.pi))

    else:
        raise ValueError('Laplace Greens kernel only implemented in 2D/3D')

    # Per component + voxel size scaling
    greens = torch.stack([greens]*3, -1)
    greens /= voxel_size.square()

    return greens


def h02(x):
    """
    Zero-th order spherical Hankel function of the second kind
    https://en.wikipedia.org/wiki/Bessel_function
    """
    j0 = x.sinc()           # zeroth spherical Bessel function
    y0 = x.cos().div_(x)    # conjugate of the zeroth spherical Neumann function
    return j0 + 1j * y0


def H01(x):
    """
    Zero-th order Hankel function of the first kind
    https://en.wikipedia.org/wiki/Bessel_function
    """
    try:
        from scipy.special import hankel1
    except ImportError:
        raise ImportError('2D Helmoltz metrics require scipy')
    device = x.device
    return torch.as_tensor(hankel1(0, x.cpu().numpy()), device=device)
