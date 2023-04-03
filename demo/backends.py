from diffeo.tests import phantoms
from diffeo.backends import (
    torch as torch_backend,
    interpol as interpol_backend,
    jitfields as jitfields_backend,
)
import torch
import interpol
import matplotlib.pyplot as plt

shape = [192, 192]
circ = phantoms.circle(shape)
letc = phantoms.letterc(shape)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(circ)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(letc)
plt.axis('off')
plt.show()

warp = torch.randn([8, 8, 2]).mul_(8)
warp = interpol.resize(
    warp.movedim(-1, 0),
    shape=shape,
    interpolation=3,
    prefilter=False
).movedim(0, -1)

warpc_itrpl = interpol_backend.pull(circ[:, :, None], warp)[:, :, 0]
warpc_torch = torch_backend.pull(circ[None, :, :, None], warp)[0, :, :, 0]
warpc_jit = jitfields_backend.pull(circ[None, :, :, None], warp)[0, :, :, 0]

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(warpc_itrpl)
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(warpc_jit)
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(warpc_torch)
plt.axis('off')
plt.show()

splatc_itrpl = interpol_backend.push(circ[:, :, None], warp)[:, :, 0]
splatc_torch = torch_backend.push(circ[None, :, :, None], warp)[0, :, :, 0]
splatc_jit = jitfields_backend.push(circ[None, :, :, None], warp)[0, :, :, 0]

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(splatc_itrpl, vmin=0, vmax=2)
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(splatc_jit, vmin=0, vmax=2)
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(splatc_torch, vmin=0, vmax=2)
plt.axis('off')
plt.show()

gradc_itrpl = interpol_backend.grad(circ[:, :, None], warp)[:, :, 0, :]
gradc_torch = torch_backend.grad(circ[None, :, :, None], warp)[0, :, :, 0, :]
gradc_jit = jitfields_backend.grad(circ[None, :, :, None], warp)[0, :, :, 0, :]

def to_rgb(x, vmin=-2, vmax=2):
    tmp = x.new_zeros([*shape, 3])
    tmp[..., :2] = x
    tmp.add_(vmin).div_(vmax-vmin).mul_(255).round_()
    tmp = tmp.to(torch.uint8)
    return tmp

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(to_rgb(gradc_itrpl))
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(to_rgb(gradc_jit))
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(to_rgb(gradc_torch))
plt.axis('off')
plt.show()

foo = 0
