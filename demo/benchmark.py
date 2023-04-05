import torch
from time import time
from diffeo.layers import Exp
import psutil
import gc


def froward_backward(v, anagrad=True):
    mem0 = psutil.Process().memory_info().rss
    # forward
    tic = time()
    exp = Exp(anagrad=anagrad)
    v.requires_grad_()
    v.grad = None
    phi = exp(v)
    tac = time() - tic
    gc.collect()
    torch.cuda.empty_cache()
    mem1 = psutil.Process().memory_info().rss
    print(f'forward | {tac} s | {(mem1-mem0)/1024**2} MB')

    # backward
    tic = time()
    phi.sum().backward()
    toc = time() - tic
    gc.collect()
    torch.cuda.empty_cache()
    mem2 = psutil.Process().memory_info().rss
    print(f'backward | {toc} s | {(mem2-mem1)/1024**2} MB')
    return phi, v.grad


v = torch.randn([1, 2, 128, 128, 128])

print('anagrad = False')
froward_backward(v, False)
froward_backward(v, False)

print('anagrad = True')
froward_backward(v, True)
froward_backward(v, True)

