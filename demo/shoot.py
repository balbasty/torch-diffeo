import torch
from diffeo.layers import Shoot, Pull, Push
from diffeo.diffdiv import diff
from diffeo.metrics import Mixture
from diffeo.tests.phantoms import circle, letterc
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def to_rgb(x):
    x = x.movedim(1, -1)
    vmin = x.min()
    vmax = x.max()
    vmax = max(abs(vmax), abs(vmin))
    vmin = -vmax
    tmp = x.new_zeros([*x.shape[:-1], 3])
    tmp[..., :x.shape[-1]] = x
    tmp.add_(vmin).div_(vmax-vmin).mul_(255).round_()
    tmp = tmp.to(torch.uint8)
    return tmp


def register(fix=None, mov=None, metric=None, hilbert=True,
             lr=0.01, nbiter=256, bound='circulant', device='cpu'):
    """Register two images by minimizing the squared differences.

    - The deformation is encoded by a geodesic.
    - Optimization uses gradient descent.

    Parameters
    ----------
    fix : (*spatial) tensor
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
        fix = letterc([192, 192], device=device, fwhm=0.02)
    if mov is None:
        mov = circle([192, 192], device=device, fwhm=0.02)
    fix, mov = fix.to(device=device), mov.to(device=device)
    fix, mov = fix.to(dtype=torch.double), mov.to(dtype=torch.double)
    if metric is None:
        metric = Mixture(
            absolute=1e-3,
            bending=0.01,
            lame_shears=1,
            lame_div=1,
            factor=1,
            use_diff=False,
            cache=True,
            bound=bound,
        )

    vel = mov.new_zeros([1, fix.ndim, *fix.shape], requires_grad=True)
    fix = fix[None, None]
    mov = mov[None, None]

    exp = Shoot(metric, fast=True)
    pull = Pull()
    alpha = 400
    max_ls = 12

    def penalty(v):
        v = v.movedim(1, -1)
        m = metric.forward(v)
        return (v*m).sum()

    for n in range(nbiter):

        vel.grad = None
        vel.requires_grad_()
        phi = exp(vel)
        wrp = pull(mov, phi)
        loss = (wrp-fix).square().sum() * alpha
        loss += penalty(vel)
        loss.backward()
        with torch.no_grad():
            if hilbert:
                vel.grad = metric.inverse(vel.grad.movedim(1, -1)).movedim(-1, 1) / 400
            ok = False
            vel.sub_(vel.grad, alpha=lr)
            for nls in range(max_ls):
                phi = exp(vel)
                wrp = pull(mov, phi)
                new_loss = (wrp-fix).square().sum() * alpha
                new_loss += penalty(vel)
                if new_loss < loss:
                    ok = True
                    lr *= 1.5
                    break
                lr /= 2
                vel.add_(vel.grad, alpha=lr)
            if max_ls and not ok:
                print('converged?')
                break

        print(f'{n:03d} | {loss.item()/mov.numel():6.3g} | lr = {lr:6.3g}', flush=True, end='\n')

        if n % 8: continue
        print('')
        show(fix, mov, wrp, vel)


def register2(fix=None, mov=None, metric=None, hilbert=True,
              lr=0.01, nbiter=1024, bound='circulant', device='cpu'):
    """SAme as above, but using JA's approximate gradients"""
    if fix is None:
        fix = letterc([192, 192], device=device, fwhm=0.02)
    if mov is None:
        mov = circle([192, 192], device=device, fwhm=0.02)
    fix, mov = fix.to(device=device), mov.to(device=device)
    fix, mov = fix.to(dtype=torch.double), mov.to(dtype=torch.double)
    if metric is None:
        metric = Mixture(
            absolute=1e-3,
            bending=0.01,
            lame_shears=1,
            lame_div=1,
            factor=1,
            use_diff=False,
            cache=True,
            bound=bound,
        )

    ndim = fix.ndim
    gradmov = diff(mov, bound='dct2', dim=range(-ndim, 0)).movedim(-1, 0)

    vel = mov.new_zeros([1, fix.ndim, *fix.shape])
    fix = fix[None, None]
    mov = mov[None, None]
    gradmov = gradmov[None]

    exp = Shoot(metric, fast=True)
    pull, push = Pull(bound='dct2'), Push()
    alpha = 400
    max_ls = 12

    def penalty(v):
        v = v.movedim(1, -1)
        m = metric.forward(v)
        return (v*m).sum()

    for n in range(nbiter):

        phi = exp(vel)
        wrp = pull(mov, phi)
        loss = (wrp-fix).square().sum() * alpha
        loss += penalty(vel)
        velgrad = push(fix-wrp, phi) * alpha
        velgrad = velgrad * gradmov
        if hilbert:
            velgrad = metric.inverse(velgrad.movedim(1, -1)).movedim(-1, 1)
            velgrad += vel
            velgrad /= 400
        else:
            velgrad += metric.forward(vel.movedim(1, -1)).movedim(-1, 1)
        ok = False
        vel.sub_(velgrad, alpha=lr)
        for nls in range(max_ls):
            phi = exp(vel)
            wrp = pull(mov, phi)
            new_loss = (wrp-fix).square().sum() * alpha
            new_loss += penalty(vel)
            if new_loss < loss:
                ok = True
                lr *= 1.5
                break
            lr /= 2
            print('try', lr)
            vel.add_(velgrad, alpha=lr)
        if max_ls and not ok:
            print('converged?')
            break

        print(f'{n:03d} | {loss.item()/mov.numel():9.6g} | lr = {lr:6.3g}', flush=True, end='\n')

        if n % 8 and (ok or max_ls == 0): continue
        print('')
        show(fix, mov, wrp, vel)

    show(fix, mov, wrp, vel)


def show(fix, mov, wrp, vel):
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
    plt.show(block=False)
    plt.pause(0.1)


register(bound='dft', device='cuda', hilbert=True, lr=1)
