import torch
from diffeo.layers import Shoot, Pull
from diffeo.metrics import Mixture
from diffeo.tests.phantoms import circle, letterc
import matplotlib.pyplot as plt


def to_rgb(x):
    x = x.movedim(1, -1)
    vmin = x.min()
    vmax = x.max()
    tmp = x.new_zeros([*x.shape[:-1], 3])
    tmp[..., :x.shape[-1]] = x
    tmp.add_(vmin).div_(vmax-vmin).mul_(255).round_()
    tmp = tmp.to(torch.uint8)
    return tmp


def register(fix=None, mov=None, metric=None, hilbert=True,
             lr=1e-2, nbiter=256, bound='circulant', device='cuda'):
    """Register two images by minimizing the squared differences.

    .. The deformation is encoded by a geodesic.
    .. Optimization uses gradient descent.

    Parameters
    ----------
    fix : (*spatial) tensorgit push --tagsgit tag -d 0.2.2
        Fixed image
    mov : (*spatial) tensor
        Moving image
    metric : diffeo.metrics.Metric
        Metric to use for regularization
    hilbert : bool
        Use Hilbert gradients.
        If True, the gradients are smoothed using the inverse of the metric.
    lr : float
        Learning rate.
    nbiter : int
        Number of gradient descent iterations.

    """
    if fix is None:
        fix = letterc([192, 192], device=device)
    if mov is None:
        mov = circle([192, 192], device=device)
    fix, mov = fix.to(device=device), mov.to(device=device)
    fix, mov = fix.to(dtype=torch.double), mov.to(dtype=torch.double)
    if metric is None:
        metric = Mixture(
            absolute=1e-4,
            membrane=1e-3,
            bending=0.2,
            lame_shears=0.05,
            lame_div=0.2,
            factor=0.1,
            use_diff=False,
            cache=True,
            bound=bound,
        )

    vel = mov.new_zeros([1, fix.ndim, *fix.shape], requires_grad=True)
    fix = fix[None, None]
    mov = mov[None, None]

    exp = Shoot(metric, fast=True)
    pull = Pull()
    max_ls = 12

    def penalty(v):
        v = v.movedim(1, -1)
        m = metric.forward(v)
        return v.flatten().dot(m.flatten())

    for n in range(nbiter):

        vel.grad = None
        phi = exp(vel)
        wrp = pull(mov, phi)
        loss = (wrp-fix).square().sum()
        loss += penalty(vel)
        loss.backward()
        with torch.no_grad():
            if hilbert:
                vel.grad = metric.inverse(vel.grad.movedim(1, -1)).movedim(-1, 1)
            ok = False
            vel.sub_(vel.grad, alpha=lr)
            for nls in range(max_ls):
                phi = exp(vel)
                wrp = pull(mov, phi)
                new_loss = (wrp-fix).square().sum()
                new_loss += penalty(vel)
                if new_loss < loss:
                    ok = True
                    lr *= 2
                    break
                lr /= 2
                vel.add_(vel.grad, alpha=lr)
            if max_ls and not ok:
                print('converged?')
                break

        print(f'{n:03d} | {loss.item()/mov.ndim:6.3g} | lr = {lr:6.3g}', end='\r')

        if n % 8: continue
        print('')
        plt.subplot(2, 2, 1)
        plt.imshow(fix.squeeze().cpu(), vmin=0, vmax=1)
        plt.axis('off')
        plt.subplot(2, 2, 2)
        plt.imshow(mov.squeeze().cpu(), vmin=0, vmax=1)
        plt.axis('off')
        plt.subplot(2, 2, 3)
        plt.imshow(wrp.detach().squeeze().cpu(), vmin=0, vmax=1)
        plt.axis('off')
        plt.subplot(2, 2, 4)
        plt.imshow(to_rgb(vel.detach()).squeeze().cpu())
        plt.axis('off')
        plt.show()


register(bound='dct2')
