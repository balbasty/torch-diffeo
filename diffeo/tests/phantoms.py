import torch
from torch.fft import fftn, ifftn, fftshift, ifftshift
from diffeo.flows import identity
import math


def circle(shape, width=0.6, fwhm=0.01, **backend):
    """Generate an image with a circle"""
    backend.setdefault('device', 'cpu')
    backend.setdefault('dtype', torch.get_default_dtype())
    width *= min(shape)

    mask = identity(shape, device=backend['device'])
    shape = torch.as_tensor(shape, dtype=mask.dtype, device=mask.device)
    mask -= (shape - 1) / 2
    mask = mask.square().sum(-1).sqrt()
    mask = mask < width/2
    mask = mask.to(backend['dtype'])
    if fwhm:
        fwhm *= min(shape)
        mask = smooth(mask, fwhm)
    return mask


def square(shape, width=0.6, fwhm=0.01, **backend):
    """Generate an image with a square"""
    backend.setdefault('device', 'cpu')
    backend.setdefault('dtype', torch.get_default_dtype())

    start = [s - int(math.floor(width * s))//2 for s in shape]
    stop = [s + s0 - int(math.floor(width * s)) for s, s0 in zip(shape, start)]
    slicer = [slice(s0 or None, (-s1) or None) for s0, s1 in zip(start, stop)]

    mask = torch.zeros(shape, **backend)
    mask[tuple(slicer)] = 1

    if fwhm:
        fwhm *= min(shape)
        mask = smooth(mask, fwhm)
    return mask


def letterc(shape, width=0.6, thickness=0.2, fwhm=0.01, **backend):
    if len(shape) != 2:
        raise NotImplementedError('Letter C only implemented in 2d')
    c = circle(shape, width, fwhm=0, **backend)
    innerc = circle(shape, width-thickness, fwhm=0,
                    dtype=torch.bool, device=c.device)
    c[innerc] = 0
    thickness *= int(math.ceil(min(shape)))
    hole = (slice(int((shape[0]-thickness)//2), int((shape[0]+thickness)//2)),
            slice(int(shape[0]//2), None))
    c[hole] = 0

    if fwhm:
        fwhm *= min(shape)
        c = smooth(c, fwhm)
    return c


def smooth(x, fwhm):
    lam = (2.355 / fwhm) ** 2
    shape = x.shape
    kernel = identity(shape, dtype=x.dtype, device=x.device)
    kernel -= torch.as_tensor(shape, dtype=x.dtype, device=x.device).sub_(1).div_(2)
    kernel = kernel.square_().sum(-1)
    kernel = kernel.mul_(-lam).exp_()
    kernel = kernel.div_(kernel.sum())

    x = fftn(ifftshift(x))
    kernel = fftn(ifftshift(kernel))
    x = x * kernel
    x = fftshift(ifftn(x)).real
    return x
