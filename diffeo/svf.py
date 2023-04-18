"""Integrate stationary velocity fields."""
__all__ = ['exp', 'bch', 'exp_forward', 'exp_backward']
import torch
from diffeo.flows import compose, compose_jacobian, jacobian, bracket
from diffeo.backends import default_backend
from diffeo.linalg import jhj, batchmatvec, batchdet

def exp(vel, steps=8, bound='circulant', anagrad=False, backend=default_backend):
    """Exponentiate a stationary velocity field by scaling and squaring.

    Parameters
    ----------
    vel : ([batch], *spatial, dim) tensor
        Stationary velocity field.
    steps : int, default=8
        Number of scaling and squaring steps
        (corresponding to 2**steps integration steps).
    bound : [list of] {'circulant', 'neumann', 'dirichlet', 'sliding'}, default='circulant'
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


def exp_forward(vel, steps=8, bound='circulant', backend=default_backend):
    vel = vel / (2**steps)
    for i in range(steps):
        vel = compose(vel, vel, bound=bound, backend=backend)
    return vel


def expjac_forward(vel, steps=8, bound='circulant', backend=default_backend):
    ndim = vel.shape[-1]
    vel = vel / (2**steps)
    jac = torch.eye(ndim, dtype=vel.dtype, device=vel.device)
    jac = jac.expand([*vel.shape[:-1], ndim, ndim])
    for i in range(steps):
        jac = compose_jacobian(jac, vel, bound=bound, backend=backend)
        vel = compose(vel, vel, bound=bound, backend=backend)
    return vel, jac


def bch(vel_left, vel_right, order=2, bound='circulant', backend=default_backend):
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


def exp_backward(vel, grad, hess=None, steps=8, bound='circulant', rotate_grad=False,
                 backend=default_backend):
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
    bound : [list of] {'circulant', 'neumann', 'dirichlet', 'sliding'}, default='circulant'
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

    def pushforward_grad(grad, vel, jac, det):
        grad = backend.pull(grad, vel, bound=bound)
        grad = batchmatvec(jac, grad)
        grad *= det[..., None]
        return grad

    def pushforward_hess(hess, vel, jac, det):
        hess = backend.pull(hess, vel, bound=bound)
        hess = jhj(jac, hess)
        hess *= det[..., None]
        return hess

    if rotate_grad:
        # It forces us to perform a forward exponentiation, which
        # is a bit annoying...
        # Maybe save the Jacobian (or phi?) after the forward pass?
        # ...but it takes space
        #
        # 2023/04/05
        # Replaced expjac with jac(exp), since the latter uses drastically
        # less memory. Might be a bit less accurate with twisted deformations.

        # _, jac = expjac_forward(vel, steps=steps, bound=bound)
        jac = exp_forward(vel, steps=steps, bound=bound)
        jac = jacobian(jac, bound=bound, add_identity=True)
        jac = jac.transpose(-1, -2)
        grad = batchmatvec(jac, grad)
        if hess is not None:
            hess = jhj(jac, hess)
        del jac

    vel = vel / (-2**steps)
    jac = jacobian(vel, bound=bound, add_identity=False)

    # rotate gradient a bit so that when steps == 0, we get the same
    # gradients as the smalldef case

    # inverse Jacobian
    jac = jac.neg_()
    jac.diagonal(0, -1, -2).add_(1)
    # rotate gradient
    grad = batchmatvec(jac.transpose(-1, -2), grad)
    # back to forward Jacobian
    jac.diagonal(0, -1, -2).sub_(1)
    jac = jac.neg_()
    jac.diagonal(0, -1, -2).add_(1)

    for i in range(steps):
        det = batchdet(jac)
        jac = jac.transpose(-1, -2)
        # pushforward, and add all scales (SVF)
        grad = pushforward_grad(grad, vel, jac, det).add_(grad)
        if hess is not None:
            hess = pushforward_hess(hess, vel, jac, det).add_(hess)
        # squaring
        jac = jac.transpose(-1, -2)
        jac = compose_jacobian(jac, vel, bound=bound, backend=backend)
        vel += backend.pull(vel, vel, bound=bound)
    del jac, det

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
        grad, _ = exp_backward(
            vel, grad,
            steps=ctx.args['steps'],
            bound=ctx.args['bound'],
            backend=ctx.args['backend'],
            rotate_grad=True,
        )
        return (grad,) + (None,)*3
