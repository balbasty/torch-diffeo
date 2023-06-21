import torch
from diffeo.utils import make_vector


def absolute(ndim, voxel_size=1, order=3, dtype=None, device=None):
    """Precision matrix for the Absolute energy, as a convolution kernel.

    Parameters
    ----------
    ndim : int
    voxel_size : float or sequence[float], default=1
    order : int, default=3

    Returns
    -------
    kernel : (ndim, order*2-1, ...) tensor

    """
    voxel_size = make_vector(voxel_size, ndim, dtype=torch.double, device='cpu').square()
    kernel = make_absolute_kernel(ndim, order)
    kernel = torch.stack([kernel]*ndim, dim=-1)
    kernel *= voxel_size
    return kernel.movedim(-1, 0).to(dtype=dtype, device=device)


def membrane(ndim, voxel_size=1, order=3, dtype=None, device=None):
    """Precision matrix for the Membrane energy, as a convolution kernel.

    Parameters
    ----------
    ndim : int
    voxel_size : float or sequence[float], default=1
    order : int, default=3

    Returns
    -------
    kernel : (ndim, order*2-1, ...) tensor

    """
    voxel_size = make_vector(voxel_size, ndim, dtype=torch.double, device='cpu').square()
    kernel = make_membrane_kernel(ndim, order, voxel_size.reciprocal())
    kernel = torch.stack([kernel]*ndim, dim=-1)
    kernel *= voxel_size
    return kernel.movedim(-1, 0).to(dtype=dtype, device=device)


def bending(ndim, voxel_size=1, order=3, dtype=None, device=None):
    """Precision matrix for the Bending energy, as a convolution kernel.

    Parameters
    ----------
    ndim : int
    voxel_size : float or sequence[float], default=1
    order : int, default=3

    Returns
    -------
    kernel : (ndim, order*2-1, ...) tensor

    """
    voxel_size = make_vector(voxel_size, ndim, dtype=torch.double, device='cpu').square()
    kernel = make_bending_kernel(ndim, order, voxel_size.reciprocal())
    kernel = torch.stack([kernel]*ndim, dim=-1)
    kernel *= voxel_size
    return kernel.movedim(-1, 0).to(dtype=dtype, device=device)


def lame_shear(ndim, voxel_size=1, order=3, dtype=None, device=None):
    """Precision matrix for the Linear Elastic energy, as a convolution kernel.

    Parameters
    ----------
    ndim : int
    voxel_size : float or sequence[float], default=1
    order : int, default=3

    Returns
    -------
    kernel : (ndim, ndim, order*2-1, ...) tensor

    """
    voxel_size = make_vector(voxel_size, ndim, dtype=torch.double, device='cpu').square()
    inv_voxel_size = voxel_size.reciprocal()
    M = make_membrane_kernel(ndim, order, inv_voxel_size)
    _, L = make_linearelastic_kernel(ndim, order, inv_voxel_size)
    kernel = torch.zeros([ndim, ndim, *M.shape], dtype=torch.double)
    c = 0
    for i in range(ndim):
        kernel[i, i] = M
        for j in range(i+1, ndim):
            kernel[i, j] = kernel[j, i] = L[c]
            c += 1
    kernel.movedim(0, -1).movedim(0, -1)
    kernel *= voxel_size[None, :] * voxel_size[:, None]
    kernel.movedim(-1, 0).movedim(-1, 0)
    return kernel.to(dtype=dtype, device=device)


def lame_div(ndim, voxel_size=1, order=3, dtype=None, device=None):
    """Precision matrix for the Linear Elastic energy, as a convolution kernel.

    Parameters
    ----------
    ndim : int
    voxel_size : float or sequence[float], default=1
    order : int, default=3

    Returns
    -------
    kernel : (ndim, ndim, order*2-1, ...) tensor

    """
    voxel_size = make_vector(voxel_size, ndim, dtype=torch.double, device='cpu').square()
    inv_voxel_size = voxel_size.reciprocal()
    K, L = make_linearelastic_kernel(ndim, order, inv_voxel_size)
    kernel = torch.zeros([ndim, ndim, *M.shape], dtype=torch.double)
    c = 0
    for i in range(ndim):
        kernel[i, i] = K
        for j in range(i+1, ndim):
            kernel[i, j] = kernel[j, i] = L[c]
            c += 1
    kernel.movedim(0, -1).movedim(0, -1)
    kernel *= voxel_size[None, :] * voxel_size[:, None]
    kernel.movedim(-1, 0).movedim(-1, 0)
    return kernel.to(dtype=dtype, device=device)


_absolute = absolute
_membrane = membrane
_bending = bending
_lame_shear = lame_shear
_lame_div = lame_div


def mixture(ndim, absolute=0, membrane=0, bending=0, lame_shear=0, lame_div=0,
            factor=1, voxel_size=1, order=3, dtype=None, device=None):
    """Precision matrix for a mixture of energies for a deformation grid.

    Parameters
    ----------
    v : (..., *spatial, dim) tensor
    absolute : float, default=0
    membrane : float, default=0
    bending : float, default=0
    lame_shear : float, default=0
    lame_div : float, default=0
    factor : float, default=1
    voxel_size : [list of] float, default=1
    bound : [list of] str, default='circulant'
    order : int, default=3

    Returns
    -------
    Lv : (..., *spatial, dim) tensor

    """
    a0 = not is_zero(absolute)
    m0 = not is_zero(membrane)
    b0 = not is_zero(bending)
    s0 = not is_zero(lame_shear)
    d0 = not is_zero(lame_div)

    voxel_size = make_vector(voxel_size, ndim, **backend)
    absolute = absolute * factor
    membrane = membrane * factor
    bending = bending * factor
    lame_shear = lame_shear * factor
    lame_div = lame_div * factor
    kopt = dict(ndim=ndim, voxel_size=voxel_size, order=order, dtype=dtype, device=device)

    kernel = 0
    if s0 or do:
        if s0:
            kernel += kernels.lame_shear(**kopt) * lame_shear
        if d0:
            kernel += kernels.lame_div(**kopt) * lame_div
        if a0:
            kernel.diagonal(0, 0, 1).add_(kernels.absolute(**kopt), alpha=absolute)
        if m0:
            kernel.diagonal(0, 0, 1).add_(kernels.membrane(**kopt), alpha=membrane)
        if b0:
            kernel.diagonal(0, 0, 1).add_(kernels.bending(**kopt), alpha=bending)
    else:
        if a0:
            kernel += kernels.absolute(**kopt) * absolute
        if m0:
            kernel += kernels.membrane(**kopt) * membrane
        if b0:
            kernel += kernels.bending(**kopt) * bending

    return kernel


def is_zero(x: Union[float, torch.Tensor]) -> bool:
    if torch.is_tensor(x) and x.requires_grad:
        return False
    else:
        return x == 0


def make_kernels1d_3():
    F0 = 151 / 315
    F1 = 397 / 1680
    F2 = 1 / 42
    F3 = 1 / 5040
    G0 = 2 / 3
    G1 = -1 / 8
    G2 = -1 / 5
    G3 = -1 / 120
    H0 = 8 / 3
    H1 = -3 / 2
    H2 = 0
    H3 = 1 / 6
    FG0 = 0
    FG1 = -49/144
    FG2 = -7/90
    FG3 = -1/720
    F = [F3, F2, F1, F0, F1, F2, F3]
    G = [G3, G2, G1, G0, G1, G2, G3]
    H = [H3, H2, H1, H0, H1, H2, H3]
    FG = [-FG3, -FG2, -FG1, FG0, FG1, FG2, FG3]
    F = torch.as_tensor(F, dtype=torch.double)
    G = torch.as_tensor(G, dtype=torch.double)
    H = torch.as_tensor(H, dtype=torch.double)
    FG = torch.as_tensor(FG, dtype=torch.double)
    return F, G, H, FG


make_kernels1d = {
    3: make_kernels1d_3,
}


def make_absolute_kernel(ndim, order=3):
    F, *_ = make_kernels1d[order]()
    K = 1
    for _ in range(ndim):
        K = K * F
        F = F[..., None]
    return K


def make_membrane_kernel(ndim, order=3, ivx2=1):
    ivx2 = make_vector(ivx2, ndim).tolist()
    F, G, *_ = make_kernels1d[order]()
    K = 0
    for i in reversed(range(ndim)):
        K1, F1, G1 = 1, F, G
        for j in reversed(range(ndim)):
            K1 = K1 * (G1 * ivx2[i] if i == j else F1)
            F1, G1 = F1[..., None], G1[..., None]
        K = K + K1
    return K


def make_bending_kernel(ndim, order=3, ivx2=1):
    ivx2 = make_vector(ivx2, ndim).tolist()
    F, G, H, *_ = make_kernels1d[order]()
    K = 0
    for i in reversed(range(ndim)):
        K1, K2, F1, G1, H1 = 1, 1, F, G, H
        for j in reversed(range(ndim)):
            K1 = K1 * (H1 * (ivx2[i] * ivx2[j]) if i == j else F1)
            K2 = K2 * (F1 if i == j else G1 * ivx2[j])
            F1, G1, H1 = F1[..., None], G1[..., None], H1[..., None]
        K = K + K1 + 2 * K2
    return K


def make_linearelastic_kernel(ndim, order=3, ivx2=1):
    ivx2 = make_vector(ivx2, ndim).tolist()
    FF, GG, _, FG = make_kernels1d[order]()
    # diagonal of lam (divergence)
    K = [1] * ndim
    for i in reversed(range(ndim)):
        FF1, GG1 = FF, GG
        for j in reversed(range(ndim)):
            K[i] = K[i] * (GG1 * ivx2[i] if i == j else FF1)
            FF1, GG1 = FF1[..., None], GG1[..., None]
    # off diagonal (common to lam and mu)
    L = [-1] * (ndim*(ndim-1)) // 2
    c = 0
    for i in range(ndim):
        for j in range(i+1, ndim):
            FF1, FG1 = FF, FG
            for k in reversed(range(ndim)):
                L[c] = L[c] * (FG1 * ivx2[k] if k in (i, j) else FF1)
                FF1, FG1 = FF1[..., None], FG1[..., None]
            c += 1
    # diagonal of mu == membrane of each component
    return tuple(K), tuple(L)

