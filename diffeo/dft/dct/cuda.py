from torch.utils.dlpack import to_dlpack, from_dlpack
import cupy
import cupyx.scipy.fft as F
import torch


flipnorm = {'forward': 'backward', 'backward': 'forward', 'ortho': 'ortho'}


class DCT(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, type, dim, norm):
        ctx.dim = dim
        ctx.norm = norm or "backward"
        ctx.type = type
        return from_cupy(F.dctn(to_cupy(x), type=type, axes=dim, norm=norm))

    @staticmethod
    def backward(ctx, x):
        norm = flipnorm[ctx.norm]
        x = from_cupy(F.idctn(to_cupy(x), type=ctx.type, axes=ctx.dim, norm=norm))
        return x, None, None, None


class IDCT(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, type, dim, norm):
        ctx.dim = dim
        ctx.norm = norm or "backward"
        ctx.type = type
        return from_cupy(F.idctn(to_cupy(x), type=type, axes=dim, norm=norm))

    @staticmethod
    def backward(ctx, x):
        norm = flipnorm[ctx.norm]
        x = from_cupy(F.dctn(to_cupy(x), type=ctx.type, axes=ctx.dim, norm=norm))
        return x, None, None, None


class DST(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, type, dim, norm):
        ctx.dim = dim
        ctx.norm = norm or "backward"
        ctx.type = type
        return from_cupy(F.dstn(to_cupy(x), type=type, axes=dim, norm=norm))

    @staticmethod
    def backward(ctx, x):
        norm = flipnorm[ctx.norm]
        x = from_cupy(F.idstn(to_cupy(x), type=ctx.type, axes=ctx.dim, norm=norm))
        return x, None, None, None


class IDST(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, type, dim, norm):
        ctx.dim = dim
        ctx.norm = norm or "backward"
        ctx.type = type
        return from_cupy(F.idstn(to_cupy(x), type=type, axes=dim, norm=norm))

    @staticmethod
    def backward(ctx, x):
        norm = flipnorm[ctx.norm]
        x = from_cupy(F.dstn(to_cupy(x), type=ctx.type, axes=ctx.dim, norm=norm))
        return x, None, None, None


def to_cupy(x):
    """Convert a torch tensor to cupy without copy"""
    return cupy.from_dlpack(to_dlpack(x))


def from_cupy(x):
    """Convert a cupy tensor to torch without copy"""
    return from_dlpack(cupy.to_dlpack(x))