from torch.utils.dlpack import to_dlpack, from_dlpack
import torch
from . import _realtransforms as F

try:
    from cupy import from_dlpack as cupy_from_dlpack, to_dlpack as cupy_to_dlpack
except ImportError:
    import cupy
    from cupy import fromDlpack as cupy_from_dlpack
    cupy_to_dlpack = cupy.ndarray.toDlpack


flipnorm = {'forward': 'backward', 'backward': 'forward', 'ortho': 'ortho'}


class DCT(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, type, dim, norm):
        ctx.dim = dim
        ctx.norm = norm or "backward"
        ctx.type = type
        return apply_cupy(F.dctn, x, type=type, axes=dim, norm=norm)

    @staticmethod
    def backward(ctx, x):
        norm = flipnorm[ctx.norm]
        x = apply_cupy(F.idctn, x, type=ctx.type, axes=ctx.dim, norm=norm)
        return x, None, None, None


class IDCT(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, type, dim, norm):
        ctx.dim = dim
        ctx.norm = norm or "backward"
        ctx.type = type
        return apply_cupy(F.idctn, x, type=type, axes=dim, norm=norm)

    @staticmethod
    def backward(ctx, x):
        norm = flipnorm[ctx.norm]
        x = apply_cupy(F.dctn, x, type=ctx.type, axes=ctx.dim, norm=norm)
        return x, None, None, None


class DST(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, type, dim, norm):
        ctx.dim = dim
        ctx.norm = norm or "backward"
        ctx.type = type
        return apply_cupy(F.dstn, x, type=type, axes=dim, norm=norm)

    @staticmethod
    def backward(ctx, x):
        norm = flipnorm[ctx.norm]
        x = apply_cupy(F.idstn, x, type=ctx.type, axes=ctx.dim, norm=norm)
        return x, None, None, None


class IDST(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, type, dim, norm):
        ctx.dim = dim
        ctx.norm = norm or "backward"
        ctx.type = type
        return apply_cupy(F.idstn, x, type=type, axes=dim, norm=norm)

    @staticmethod
    def backward(ctx, x):
        norm = flipnorm[ctx.norm]
        x = apply_cupy(F.dstn, x, type=ctx.type, axes=ctx.dim, norm=norm)
        return x, None, None, None


def to_cupy(x):
    """Convert a torch tensor to cupy without copy"""
    return cupy_from_dlpack(to_dlpack(x))


def from_cupy(x):
    """Convert a cupy tensor to torch without copy"""
    return from_dlpack(cupy_to_dlpack(x))


def apply_cupy(func, x, *args, **kwargs):
    """Manipulate tensor to apply a cupy function"""
    if x.is_complex():
        return (apply_cupy(func, x.real, *args, **kwargs) +
                apply_cupy(func, x.real, *args, **kwargs) * 1j)
    return from_cupy(func(to_cupy(x), *args, **kwargs))
