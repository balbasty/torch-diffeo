"""
Linear Algebra utilities.

I found that some torch functions (e.g., `inverse()` or `det()`) where
not so efficient when applied to large batches of small matrices,
especially on the GPU (this is not so true on the CPU). I reimplemented
them using torchscript for 2x2 and 3x3 matrices, and they are much
faster:
    - batchdet
    - batchinv
    - batchmatvec
I used to have a `batchmatmul` too, but its speed was not always better
than `torch.matmul()` (it depended a lot on the striding layout),
so I removed it.
"""

import torch


def matvec(mat, vec):
    return mat.matmul(vec.unsqueeze(-1)).squeeze(-1)


@torch.jit.script
def det2(A):
    dt = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    return dt


@torch.jit.script
def det3(A):
    dt = A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]) + \
         A[0, 1] * (A[1, 2] * A[2, 0] - A[1, 0] * A[2, 2]) + \
         A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0])
    return dt


def batchdet(A):
    if not A.is_cuda:
        return A.det()
    A = A.movedim(-1, 0).movedim(-1, 0)
    if len(A) == 3:
        A = det3(A)
    elif len(A) == 2:
        A = det2(A)
    else:
        assert len(A) == 1
        A = A.clone()[0, 0]
    return A


@torch.jit.script
def inv2(A):
    F = torch.empty_like(A)
    F[0, 0] = A[1, 1]
    F[1, 1] = A[0, 0]
    F[0, 1] = -A[1, 0]
    F[1, 0] = -A[0, 1]
    dt = det2(A)
    Aabs = A.reshape((-1,) + A.shape[2:]).abs()
    rnge = Aabs.max(dim=0).values - Aabs.min(dim=0).values
    dt += rnge * 1E-12
    F /= dt[None, None]
    return F


@torch.jit.script
def inv3(A):
    F = torch.empty_like(A)
    F[0, 0] = A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]
    F[1, 1] = A[0, 0] * A[2, 2] - A[0, 2] * A[2, 0]
    F[2, 2] = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    F[0, 1] = A[0, 2] * A[2, 1] - A[0, 1] * A[2, 2]
    F[0, 2] = A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1]
    F[1, 0] = A[1, 2] * A[2, 0] - A[1, 0] * A[2, 2]
    F[1, 2] = A[1, 0] * A[0, 2] - A[1, 2] * A[0, 0]
    F[2, 0] = A[2, 1] * A[1, 0] - A[2, 0] * A[1, 1]
    F[2, 1] = A[2, 0] * A[0, 1] - A[2, 1] * A[0, 0]
    dt = det3(A)
    Aabs = A.reshape((-1,) + A.shape[2:]).abs()
    rnge = Aabs.max(dim=0).values - Aabs.min(dim=0).values
    dt += rnge * 1E-12
    F /= dt[None, None]
    return F


def batchinv(A):
    if not A.is_cuda:
        return A.inverse()
    A = A.movedim(-1, 0).movedim(-1, 0)
    if len(A) == 3:
        A = inv3(A)
    elif len(A) == 2:
        A = inv2(A)
    else:
        assert len(A) == 1
        A = A.reciprocal()
    A = A.movedim(0, -1).movedim(0, -1)
    return A


@torch.jit.script
def jhj1(jac, hess):
    # jac should be ordered as (D, ...)
    # hess should be ordered as (D, D, ...)
    return jac[0, 0] * jac[0, 0] * hess[0]


@torch.jit.script
def jhj2(jac, hess):
    # jac should be ordered as (D, ...)
    # hess should be ordered as (D, D, ...)
    out = torch.empty_like(hess)
    h00 = hess[0]
    h11 = hess[1]
    h01 = hess[2]
    j00 = jac[0, 0]
    j01 = jac[0, 1]
    j10 = jac[1, 0]
    j11 = jac[1, 1]
    out[0] = j00 * j00 * h00 + j01 * j01 * h11 + 2 * j00 * j01 * h01
    out[1] = j10 * j10 * h00 + j11 * j11 * h11 + 2 * j10 * j11 * h01
    out[2] = j00 * j10 * h00 + j01 * j11 * h11 + (j01 * j10 + j00 * j11) * h01
    return out


@torch.jit.script
def jhj3(jac, hess):
    # jac should be ordered as (D, ...)
    # hess should be ordered as (D, D, ...)
    out = torch.empty_like(hess)
    h00 = hess[0]
    h11 = hess[1]
    h22 = hess[2]
    h01 = hess[3]
    h02 = hess[4]
    h12 = hess[5]
    j00 = jac[0, 0]
    j01 = jac[0, 1]
    j02 = jac[0, 2]
    j10 = jac[1, 0]
    j11 = jac[1, 1]
    j12 = jac[1, 2]
    j20 = jac[2, 0]
    j21 = jac[2, 1]
    j22 = jac[2, 2]
    out[0] = h00 * j00 * j00 + 2 * h01 * j00 * j01 + 2 * h02 * j00 * j02 + h11 * j01 * j01 + 2 * h12 * j01 * j02 + h22 * j02 * j02
    out[1] = h00 * j10 * j10 + 2 * h01 * j10 * j11 + 2 * h02 * j10 * j12 + h11 * j11 * j11 + 2 * h12 * j11 * j12 + h22 * j12 * j12
    out[2] = h00 * j20 * j20 + 2 * h01 * j20 * j21 + 2 * h02 * j20 * j22 + h11 * j21 * j21 + 2 * h12 * j21 * j22 + h22 * j22 * j22
    out[3] = j10 * (h00 * j00 + h01 * j01 + h02 * j02) + j11 * (h01 * j00 + h11 * j01 + h12 * j02) + j12 * (h02 * j00 + h12 * j01 + h22 * j02)
    out[4] = j20 * (h00 * j00 + h01 * j01 + h02 * j02) + j21 * (h01 * j00 + h11 * j01 + h12 * j02) + j22 * (h02 * j00 + h12 * j01 + h22 * j02)
    out[5] = j20 * (h00 * j10 + h01 * j11 + h02 * j12) + j21 * (h01 * j10 + h11 * j11 + h12 * j12) + j22 * (h02 * j10 + h12 * j11 + h22 * j12)
    return out


def jhj(jac, hess):
    """J*H*J', where H is symmetric and stored sparse"""

    # Matlab symbolic toolbox
    #
    # 2D:
    # out[00] = h00*j00^2 + h11*j01^2 + 2*h01*j00*j01
    # out[11] = h00*j10^2 + h11*j11^2 + 2*h01*j10*j11
    # out[01] = h00*j00*j10 + h11*j01*j11 + h01*(j01*j10 + j00*j11)
    #
    # 3D:
    # out[00] = h00*j00^2 + 2*h01*j00*j01 + 2*h02*j00*j02 + h11*j01^2 + 2*h12*j01*j02 + h22*j02^2
    # out[11] = h00*j10^2 + 2*h01*j10*j11 + 2*h02*j10*j12 + h11*j11^2 + 2*h12*j11*j12 + h22*j12^2
    # out[22] = h00*j20^2 + 2*h01*j20*j21 + 2*h02*j20*j22 + h11*j21^2 + 2*h12*j21*j22 + h22*j22^2
    # out[01] = j10*(h00*j00 + h01*j01 + h02*j02) + j11*(h01*j00 + h11*j01 + h12*j02) + j12*(h02*j00 + h12*j01 + h22*j02)
    # out[02] = j20*(h00*j00 + h01*j01 + h02*j02) + j21*(h01*j00 + h11*j01 + h12*j02) + j22*(h02*j00 + h12*j01 + h22*j02)
    # out[12] = j20*(h00*j10 + h01*j11 + h02*j12) + j21*(h01*j10 + h11*j11 + h12*j12) + j22*(h02*j10 + h12*j11 + h22*j12)

    ndim = jac.shape[-1]
    hess = hess.movedim(-1, 0)
    jac = jac.movedim(-1, 0).movedim(-1, 0)
    if ndim == 1:
        out = jhj1(jac, hess)
    elif ndim == 2:
        out = jhj2(jac, hess)
    else:
        assert ndim == 3
        out = jhj3(jac, hess)
    out = out.movedim(0, -1)
    return out


@torch.jit.script
def matvec3(A, v):
    Av = torch.empty_like(v)
    Av[0] = A[0, 0] * v[0] + A[0, 1] * v[1] + A[0, 2] * v[2]
    Av[1] = A[1, 0] * v[0] + A[1, 1] * v[1] + A[1, 2] * v[2]
    Av[2] = A[2, 0] * v[0] + A[2, 1] * v[1] + A[2, 2] * v[2]
    return Av


@torch.jit.script
def matvec2(A, v):
    Av = torch.empty_like(v)
    Av[0] = A[0, 0] * v[0] + A[0, 1] * v[1]
    Av[1] = A[1, 0] * v[0] + A[1, 1] * v[1]
    return Av


@torch.jit.script
def matvec1(A, v):
    Av = torch.empty_like(v)
    Av[0] = A[0, 0] * v[0]
    return Av


def batchmatvec(mat, vec):
    if not mat.is_cuda:
        return matvec(mat, vec)
    ndim = mat.shape[-1]
    vec = vec.movedim(-1, 0)
    mat = mat.movedim(-1, 0).movedim(-1, 0)
    if ndim == 1:
        mv = matvec1
    elif ndim == 2:
        mv = matvec2
    else:
        assert ndim == 3
        mv = matvec3
    out = mv(mat, vec)
    out = out.movedim(0, -1)
    return out
