"""Integrate stationary velocity fields."""
__all__ = ['exp', 'bch', 'exp_forward', 'exp_backward']
import torch
from diffeo.flows import compose, compose_jacobian, jacobian, bracket
from diffeo.backends import interpol
from diffeo.linalg import matvec, jhj


def exp(vel, steps=8, bound='dft', anagrad=False, backend=interpol):
    """Exponentiate a stationary velocity field by scaling and squaring.

    Parameters
    ----------
    vel : ([batch], *spatial, dim) tensor
        Stationary velocity field.
    steps : int, default=8
        Number of scaling and squaring steps
        (corresponding to 2**steps integration steps).
    bound : str, default='dft'
        Boundary conditions
    anagrad : bool, default=False
        Use analytical gradients rather than autodiff gradients in
        the backward pass. Should be more memory efficient and (maybe)
        faster.

    Returns
    -------
    grid : ([batch], *spatial, dim) tensor
        Exponentiated displacement

    """
    exp_fn = _Exp.apply if anagrad else exp_forward
    flow = exp_fn(vel, steps, bound, backend)
    return flow


def exp_forward(vel, steps=8, bound='dft', backend=interpol):
    vel = vel / (2**steps)
    for i in range(steps):
        vel = compose(vel, vel, bound=bound, backend=backend)
    return vel


def expjac_forward(vel, steps=8, bound='dft', backend=interpol):
    ndim = vel.shape[-1]
    vel = vel / (2**steps)
    jac = torch.eye(ndim, dtype=vel.dtype, device=vel.device)
    jac = jac.expand([*vel.shape[:-1], ndim, ndim])
    for i in range(steps):
        jac = compose_jacobian(jac, vel, bound=bound, backend=backend)
        vel = compose(vel, vel, bound=bound, backend=backend)
    return vel, jac


def bch(vel_left, vel_right, order=2, bound='dft', backend=interpol):
    """Find v such that exp(v) = exp(u) o exp(w) using the
    (truncated) Baker–Campbell–Hausdorff formula.

    https://en.wikipedia.org/wiki/BCH_formula

    Parameters
    ----------
    vel_left : (B, *shape, D) tensor
    vel_right : (B, *shape, D) tensor
    order : 1..4, default=2
        Truncation order.

    Returns
    -------
    vel : (B, *shape, D) tensor

    """
    brkt = lambda a, b: bracket(a, b, bound=bound, backend=backend)
    vel = vel_left + vel_right
    if order > 1:
        b1 = brkt(vel_left, vel_right)
        vel.add_(b1, alpha=1/2)
        if order > 2:
            b2_left = brkt(vel_left, b1)
            vel.add_(b2_left, alpha=1/12)
            b2_right = brkt(vel_right, b1)
            vel.add_(b2_right, alpha=-1/12)
            if order > 3:
                b3 = brkt(vel_right, b2_left)
                vel.add_(b3, alpha=-1/24)
                if order > 4:
                    raise ValueError('BCH only implemented up to order 4')
    return vel


def exp_backward(vel, grad, hess=None, steps=8, bound='dft', rotate_grad=False,
                 backend=interpol):
    """Backward pass of SVF exponentiation.

    This should be much more memory-efficient than the autograd pass
    as we don't have to store intermediate grids.

    I am using DARTEL's derivatives (from the code, not the paper).
    From what I get, it corresponds to pushing forward the gradient
    (computed in observation space) recursively while squaring the
    (inverse) transform.
    Remember that the push forward of g by phi is
                    |iphi| iphi' * g(iphi)
    where iphi is the inverse of phi. We could also have implemented
    this operation as: inverse(phi)' * push(g, phi), since
    push(g, phi) \approx |iphi| g(iphi). It has the advantage of using
    push rather than pull, which might preserve better positive-definiteness
    of the Hessian, but requires the inversion of (potentially ill-behaved)
    Jacobian matrices.

    Note that gradients must first be rotated using the Jacobian of
    the exponentiated transform so that the denominator refers to the
    initial velocity (we want dL/dV0, not dL/dPsi).
    THIS IS NOT DONE INSIDE THIS FUNCTION YET (see _dartel).

    Parameters
    ----------
    vel : (..., *spatial, dim) tensor
        Velocity
    grad : (..., *spatial, dim) tensor
        Gradient with respect to the output grid
    hess : (..., *spatial, dim*(dim+1)//2) tensor, optional
        Symmetric hessian with respect to the output grid.
    steps : int, default=8
        Number of scaling and squaring steps
    bound : str, default='dft'
        Boundary condition
    rotate_grad : bool, default=False
        If True, rotate the gradients using the Jacobian of exp(vel).

    Returns
    -------
    grad : (..., *spatial, dim) tensor
        Gradient with respect to the SVF
    hess : (..., *spatial, dim*(dim+1)//2) tensor, optional
        Approximate (block diagonal) Hessian with respect to the SVF

    """
    ndim = vel.shape[-1]

    if rotate_grad:
        # It forces us to perform a forward exponentiation, which
        # is a bit annoying...
        # Maybe save the Jacobian after the forward pass? But it take space
        _, jac = expjac_forward(vel, steps=steps, bound=bound)
        jac = jac.transpose(-1, -2)
        grad = matvec(jac, grad)
        if hess is not None:
            hess = jhj(jac, hess)
        del jac

    vel = vel / (-2**steps)
    jac = jacobian(vel, bound=bound, add_identity=True)

    # rotate gradient a bit so that when steps == 0, we get the same
    # gradients as the smalldef case
    ijac = 2 * torch.eye(ndim, dtype=jac.dtype, device=jac.device) - jac
    ijac = ijac.transpose(-1, -2).inverse()
    grad = matvec(ijac, grad)
    del ijac

    for _ in range(steps):
        det = jac.det()
        jac = jac.transpose(-1, -2)
        grad0 = grad
        grad = backend.pull(grad, vel, bound=bound)  # \
        grad = matvec(jac, grad)                     # | push forward
        grad *= det[..., None]                       # /
        grad += grad0                                # add all scales (SVF)
        if hess is not None:
            hess0 = hess
            hess = backend.pull(hess, vel, bound=bound)
            hess = jhj(jac, hess)
            hess *= det[..., None]
            hess += hess0
        # squaring
        jac = jac.transpose(-1, -2)
        jac = compose_jacobian(jac, vel, bound=bound, backend=backend)
        vel += backend.pull(vel, vel, bound=bound)

    grad /= (2**steps)
    if hess is not None:
        hess /= (2**steps)

    return grad, hess


class _Exp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, vel, steps, bound, backend):
        if vel.requires_grad:
            ctx.save_for_backward(vel)
            ctx.args = {'steps': steps,  'bound': bound, 'backend': backend}
        return exp_forward(vel, steps, bound, backend)

    @staticmethod
    def backward(ctx, grad):
        vel, = ctx.saved_tensors
        grad = exp_backward(vel, grad,
                            steps=ctx.args['steps'],
                            bound=ctx.args['bound'],
                            backend=ctx.args['backend'],
                            rotate_grad=True)
        return (grad,) + (None,)*3
