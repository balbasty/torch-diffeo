import torch
from diffeo.layers import Exp, Pull
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


def register(fix=None, mov=None, metric=None, hilbert=True, lr=1e-3, nbiter=256):
    """Register two images by minimizing the squared differences.

    .. The deformation is encoded by a stationary velocity field.
    .. Optimization uses gradient descent.

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
        fix = letterc([192, 192])
    if mov is None:
        mov = circle([192, 192])
    if metric is None:
        metric = Mixture(
            absolute=1e-3,
            lame_shears=0.1,
            lame_div=1e-3,
            factor=1e-2,
            use_diff=True)

    vel = mov.new_zeros([1, fix.ndim, *fix.shape], requires_grad=True)
    fix = fix[None, None]
    mov = mov[None, None]

    exp = Exp(anagrad=True)
    pull = Pull()

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
            vel.sub_(vel.grad, alpha=lr)

        print(f'{n:03d} | {loss.item()/mov.ndim:6.3g}', end='\r')

        if n % 16: continue
        print('')
        plt.subplot(2, 2, 1)
        plt.imshow(fix.squeeze(), vmin=0, vmax=1)
        plt.axis('off')
        plt.subplot(2, 2, 2)
        plt.imshow(mov.squeeze(), vmin=0, vmax=1)
        plt.axis('off')
        plt.subplot(2, 2, 3)
        plt.imshow(wrp.detach().squeeze(), vmin=0, vmax=1)
        plt.axis('off')
        plt.subplot(2, 2, 4)
        plt.imshow(to_rgb(vel.detach()).squeeze())
        plt.axis('off')
        plt.show()


register()
