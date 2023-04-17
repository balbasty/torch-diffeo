import torch
from time import time
from diffeo.layers import Exp
from diffeo.backends import (
    torch as torch_backend,
    interpol as interpol_backend,
    jitfields as jitfields_backend,
)
import psutil
import gc


def froward_backward(v, anagrad=True, backend=jitfields_backend):
    torch.cuda.reset_max_memory_allocated()
    if v.is_cuda:
        mema0 = torch.cuda.memory_allocated()
        memp0 = torch.cuda.max_memory_allocated()
        print(f'init     | {0:7.3g} s | '
              f'{mema0/1024**2:6.2f} MB | max = {memp0/1024**2:6.2f} MB ')
    else:
        mem0 = psutil.Process().memory_info().rss
        print(f'init     | {0:7.3g} s | {mem0/1024**2:6.2f} MB')
    # forward
    torch.cuda.reset_max_memory_allocated()
    tic = time()
    exp = Exp(anagrad=anagrad, backend=backend)
    v.requires_grad_()
    v.grad = None
    phi = exp(v)
    tac = time() - tic
    gc.collect()
    torch.cuda.empty_cache()
    if v.is_cuda:
        mema1 = torch.cuda.memory_allocated()
        memp1 = torch.cuda.max_memory_allocated()
        print(f'forward  | {tac:7.3g} s | '
              f'{mema1/1024**2:6.2f} MB | max = {memp1/1024**2:6.2f} MB ')
    else:
        mem1 = psutil.Process().memory_info().rss
        print(f'forward  | {tac:7.3g} s | {mem1/1024**2:6.2f} MB')

    # backward
    torch.cuda.reset_max_memory_allocated()
    tic = time()
    phi.sum().backward()
    toc = time() - tic
    gc.collect()
    torch.cuda.empty_cache()
    if v.is_cuda:
        mema2 = torch.cuda.memory_allocated()
        memp2 = torch.cuda.max_memory_allocated()
        print(f'backward | {toc:7.3g} s | '
              f'{mema2/1024**2:6.2f} MB | max = {memp2/1024**2:6.2f} MB ')
    else:
        mem2 = psutil.Process().memory_info().rss
        print(f'backward | {toc:7.3g} s | {mem2/1024**2:6.2f} MB')
    return phi, v.grad


v = torch.randn([1, 3, 128, 128, 128])

for backend in (jitfields_backend, interpol_backend, torch_backend):
    print('')
    print('====', backend.__name__, '====')
    for device in ('cpu', 'cuda'):
        print('----', device, '----')
        v = v.to(device)
        for anagrad in (False, True):
            print('anagrad =', anagrad)
            froward_backward(v, anagrad, backend=backend)
            froward_backward(v, anagrad, backend=backend)
            froward_backward(v, anagrad, backend=backend)
        print('')
