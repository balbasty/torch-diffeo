__all__ = [
    'dct', 'dct1', 'dct2', 'dct3', 'dct4',
    'dst', 'dst1', 'dst2', 'dst3', 'dst4',
    'idct', 'idct1', 'idct2', 'idct3', 'idct4',
    'idst', 'idst1', 'idst2', 'idst3', 'idst4'
]

import torch
from . import cpu
if torch.cuda.is_available():
    from . import cuda
else:
    cuda = None


def dct(x, dim=-1, norm=None, type=2):
    if type == 1:
        return dct1(x, dim, norm)
    elif type == 2:
        return dct2(x, dim, norm)
    elif type == 3:
        return dct3(x, dim, norm)
    elif type == 4:
        return dct4(x, dim, norm)
    else:
        raise ValueError('DCT only implemented for types I-IV')


def idct(x, dim=-1, norm=None, type=2):
    if type == 1:
        return idct1(x, dim, norm)
    elif type == 2:
        return idct2(x, dim, norm)
    elif type == 3:
        return idct3(x, dim, norm)
    elif type == 4:
        return idct4(x, dim, norm)
    else:
        raise ValueError('IDCT only implemented for types I-IV')


def dst(x, dim=-1, norm=None, type=2):
    if type == 1:
        return dst1(x, dim, norm)
    elif type == 2:
        return dst2(x, dim, norm)
    elif type == 3:
        return dst3(x, dim, norm)
    elif type == 4:
        return dst4(x, dim, norm)
    else:
        raise ValueError('DCT only implemented for types I-IV')


def idst(x, dim=-1, norm=None, type=2):
    if type == 1:
        return idst1(x, dim, norm)
    elif type == 2:
        return idst2(x, dim, norm)
    elif type == 3:
        return idst3(x, dim, norm)
    elif type == 4:
        return idst4(x, dim, norm)
    else:
        raise ValueError('DCT only implemented for types I-IV')


def dct1(x, dim=-1, norm=None):
    DCT = cuda.DCT if x.is_cuda else cpu.DCT
    return DCT.apply(x, 1, dim, norm)


def idct1(x, dim=-1, norm=None):
    IDCT = cuda.IDCT if x.is_cuda else cpu.IDCT
    return IDCT.apply(x, 1, dim, norm)


def dct2(x, dim=-1, norm=None):
    DCT = cuda.DCT if x.is_cuda else cpu.DCT
    return DCT.apply(x, 2, dim, norm)


def idct2(x, dim=-1, norm=None):
    IDCT = cuda.IDCT if x.is_cuda else cpu.IDCT
    return IDCT.apply(x, 2, dim, norm)


def dct3(x, dim=-1, norm=None):
    DCT = cuda.DCT if x.is_cuda else cpu.DCT
    return DCT.apply(x, 3, dim, norm)


def idct3(x, dim=-1, norm=None):
    IDCT = cuda.IDCT if x.is_cuda else cpu.IDCT
    return IDCT.apply(x, 3, dim, norm)


def dct4(x, dim=-1, norm=None):
    DCT = cuda.DCT if x.is_cuda else cpu.DCT
    return DCT.apply(x, 3, dim, norm)


def idct4(x, dim=-1, norm=None):
    IDCT = cuda.IDCT if x.is_cuda else cpu.IDCT
    return IDCT.apply(x, 3, dim, norm)


def dst1(x, dim=-1, norm=None):
    DST = cuda.DST if x.is_cuda else cpu.DST
    return DST.apply(x, 1, dim, norm)


def idst1(x, dim=-1, norm=None):
    IDST = cuda.IDST if x.is_cuda else cpu.IDST
    return IDST.apply(x, 1, dim, norm)


def dst2(x, dim=-1, norm=None):
    DST = cuda.DST if x.is_cuda else cpu.DST
    return DST.apply(x, 2, dim, norm)


def idst2(x, dim=-1, norm=None):
    IDST = cuda.IDST if x.is_cuda else cpu.IDST
    return IDST.apply(x, 2, dim, norm)


def dst3(x, dim=-1, norm=None):
    DST = cuda.DST if x.is_cuda else cpu.DST
    return DST.apply(x, 3, dim, norm)


def idst3(x, dim=-1, norm=None):
    IDST = cuda.IDST if x.is_cuda else cpu.IDST
    return IDST.apply(x, 3, dim, norm)


def dst4(x, dim=-1, norm=None):
    DST = cuda.DST if x.is_cuda else cpu.DST
    return DST.apply(x, 3, dim, norm)


def idst4(x, dim=-1, norm=None):
    IDST = cuda.IDST if x.is_cuda else cpu.IDST
    return IDST.apply(x, 3, dim, norm)
