import scipy.fft as F
import torch


flipnorm = {'forward': 'backward', 'backward': 'forward', 'ortho': 'ortho'}


class DCT(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, type, dim, norm):
        ctx.dim = dim
        ctx.norm = norm or "backward"
        ctx.type = type
        return F.dctn(x, type=type, axes=dim, norm=norm)

    @staticmethod
    def backward(ctx, x):
        norm = flipnorm[ctx.norm]
        x = F.idctn(x, type=ctx.type, axes=ctx.dim, norm=norm)
        return x, None, None, None


class IDCT(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, type, dim, norm):
        ctx.dim = dim
        ctx.norm = norm or "backward"
        ctx.type = type
        return F.idctn(x, type=type, axes=dim, norm=norm)

    @staticmethod
    def backward(ctx, x):
        norm = flipnorm[ctx.norm]
        x = F.dctn(x, type=ctx.type, axes=ctx.dim, norm=norm)
        return x, None, None, None


class DST(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, type, dim, norm):
        ctx.dim = dim
        ctx.norm = norm or "backward"
        ctx.type = type
        return F.dstn(x, type=type, axes=dim, norm=norm)

    @staticmethod
    def backward(ctx, x):
        norm = flipnorm[ctx.norm]
        x = F.idstn(x, type=ctx.type, axes=ctx.dim, norm=norm)
        return x, None, None, None


class IDST(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, type, dim, norm):
        ctx.dim = dim
        ctx.norm = norm or "backward"
        ctx.type = type
        return F.idstn(x, type=type, axes=dim, norm=norm)

    @staticmethod
    def backward(ctx, x):
        norm = flipnorm[ctx.norm]
        x = F.dstn(x, type=ctx.type, axes=ctx.dim, norm=norm)
        return x, None, None, None
