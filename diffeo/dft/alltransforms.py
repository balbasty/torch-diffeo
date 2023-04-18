__all__ = [
    'dft', 'idft',
    'dct', 'dct1', 'dct2', 'dct3', 'dct4',
    'dst', 'dst1', 'dst2', 'dst3', 'dst4',
    'idct', 'idct1', 'idct2', 'idct3', 'idct4',
    'idst', 'idst1', 'idst2', 'idst3', 'idst4',
]

import torch
try:
    from .dct import *
except ImportError:
    dct = dct1 = dct2 = dct3 = dct4 = idct = idct1 = idct2 = idct3 = idct4 = None
    dst = dst1 = dst2 = dst3 = dst4 = idst = idst1 = idst2 = idst3 = idst4 = None

dft = torch.fft.fftn
idft = torch.fft.ifftn
