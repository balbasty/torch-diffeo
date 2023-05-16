import torch
import torch.nn.functional as F
from .utils import ensure_list
from .bounds import has_sliding, sliding2dft, bound2dft
from .padding import pad


def flowconv(flow, kernel, bound='dft'):
    """Convolution with a regularization kernel

    Parameters
    ----------
    flow : (..., *spatial, ndim) tensor
        Input flow field.
    kernel : ([[ndim], ndim], *kershape) tensor
        Convolution kernel.
        - If shape `(*kershape)`, apply the same kernel to all channels
        - If shape `(inchannels, *kershape)`, apply a different kernel
        per channel
        - If shape `(outchannels, inchannels, *kershape)`, apply a full
        kernel that mixes channels.
    bound : [list of] str
        Boundary conditions used for padding.

    Returns
    -------
    out : (..., *spatial, ndim) tensor
        Output flow field.

    """
    ndim = flow.shape[-1]
    flow = flow.movedim(-1, -ndim-1)
    flow = conv(ndim, flow, kernel, padding='same', bound=bound)
    flow = flow.movedim(-ndim-1, -1)
    return flow


def conv(ndim, inp, kernel, padding=0, bound='zero'):
    """Convolution with a regularization kernel

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions
    inp : (..., [inchannels], *spatial) tensor
        Input image.
    kernel : ([[outchannels], inchannels], *kershape) tensor
        Convolution kernel.
        - If shape `(*kershape)`, apply the same kernel to all channels
        - If shape `(inchannels, *kershape)`, apply a different kernel
        per channel
        - If shape `(outchannels, inchannels, *kershape)`, apply a full
        kernel that mixes channels.
    padding : int or 'same'
        Padding size.
    bound : [list of] str
        Boundary conditions used for padding.

    Returns
    -------
    out : (..., outchannels, *spatial) tensor
        Output image.

    """
    batch = inp.shape[:-ndim-1]
    inchannels = inp.shape[-ndim-1] if inp.ndim > ndim else 1
    if kernel.ndim not in (ndim, ndim+1, ndim+2):
        spatial = ('W, H, D' if ndim == 3 else
                   'W, H' if ndim == 2 else
                   'W' if ndim == 1 else
                   '*spatial')
        raise ValueError(f'Kernel shape should be ({spatial}) or '
                         f'(Cin, {spatial}) or (Cout, Cin, {spatial})')
    if kernel.ndim > ndim and kernel.shape[-ndim-1] != inchannels:
        raise ValueError('Number of input channels in image and kernel '
                         'are not consistent')
    outchannels = kernel.shape[-ndim-2] if kernel.ndim == ndim+2 else inchannels
    kershape = kernel.shape[-ndim:]
    inshape = inp.shape[-ndim:]

    padding = _make_padding(padding, kershape)
    bigshape = torch.Size([s+2*p for s, p in zip(inshape, padding)])
    bound = _make_bound(bound, ndim)
    if has_sliding(bound):
        if inchannels != ndim or outchannels != ndim:
            raise ValueError('"sliding" boundary can only be used on flow fields')
        inp0 = inp.movedim(-ndim-1, 0)
        inp = inp0.new_empty(inp0.shape[:-ndim] + bigshape).movedim(-ndim-1, 0)
        for d, inp1 in enumerate(inp0):
            bound1 = sliding2dft(bound, d)
            inp[d] = pad(inp, padding, mode=bound1, side='both')
        inp = inp.movedim(0, -ndim-1)
    else:
        inp = pad(inp, padding, mode=bound, side='both')

    conv_fn = getattr(F, f'conv{ndim}d')
    if kernel.ndim == ndim:
        # shared across channels
        inp = inp.reshape([-1, 1, *bigshape])
        out = conv_fn(inp, kernel[None, None])
    elif kernel.ndim == ndim + 1:
        # diagonal kernel
        inp = inp.reshape([-1, inchannels, *bigshape])
        out = conv_fn(inp, kernel[:, None], groups=inchannels)
    else:
        # full kernel
        inp = inp.reshape([-1, inchannels, *bigshape])
        out = conv_fn(inp, kernel)
    out = out.reshape([*batch, outchannels, *out.shape[-ndim:]])
    return out


def _make_padding(padding, kershape):
    padding = ensure_list(padding, len(kershape))
    new_padding = []
    for p, s in zip(padding, kershape):
        if p == 'same':
            new_padding.append((s-1)//2)
        else:
            new_padding.append(p)
    return new_padding


def _make_bound(bound, ndim):
    bound = ensure_list(bound, ndim)
    bound = list(map(lambda x: bound2dft.get(x, x), bound))
    return bound
