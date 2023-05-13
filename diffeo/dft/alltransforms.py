__all__ = [
    'dft', 'idft',
    'dct', 'dct1', 'dct2', 'dct3', 'dct4',
    'dst', 'dst1', 'dst2', 'dst3', 'dst4',
    'idct', 'idct1', 'idct2', 'idct3', 'idct4',
    'idst', 'idst1', 'idst2', 'idst3', 'idst4',
]

import torch
from .realtransforms import *

dft = torch.fft.fftn
idft = torch.fft.ifftn
