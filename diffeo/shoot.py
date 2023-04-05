"""Geodesic shooting of initial velocity fields."""
import torch
from diffeo.metrics.discrete import Mixture
from diffeo.backends import interpol
from diffeo.linalg import batchmatvec, batchinv
from diffeo.flows import jacobian, compose


default_metric = Mixture(absolute=0.001, membrane=0.1)


def shoot(vel, metric=default_metric, steps=8, fast=True, verbose=False,
          backend=interpol):
    """Exponentiate a velocity field by geodesic shooting.

    Notes
    -----
    In our convention, the initial velocity is defined in the space
    of the moving image. It is the opposite of the LDDMM convention,
    which defines the initial velocity in the space of the fixed image.
    Our phi is therefore LDDMM's iphi, and vice versa.

    Parameters
    ----------
    vel : (..., *spatial, dim) tensor
        Initial velocity in moving space.
    metric : Metric
        Riemannian metric
    steps : int, default=8
        Number of integration steps.
        If None, use an educated guess based on the magnitude of `vel`.
    fast : bool, default=True
        If True, use a faster integration scheme, which may induce
        some numerical error (the energy is not exactly preserved
        across time). Else, use the slower but more precise scheme.

    Returns
    -------
    flow : (..., *spatial, dim) tensor
        Transformation from fixed to moving space.
        (It is used to warp a moving image to a fixed one).

    iflow : (..., *spatial, dim) tensor, if return_inverse
        Inverse transformation, from fixed to moving space.
        (It is used to warp a fixed image to a moving one).

    """
    # Authors
    # -------
    # .. John Ashburner <j.ashburner@ucl.ac.uk> : original Matlab code
    # .. Yael Balbastre <yael.balbastre@gmail.com> : Python port
    #
    # License
    # -------
    # The original Matlab code is (C) 2012-2019 WCHN / John Ashburner
    # and was distributed as part of [SPM](https://www.fil.ion.ucl.ac.uk/spm)
    # under the GNU General Public Licence (version >= 2).

    ndim = vel.shape[-1]
    spatial = vel.shape[-ndim-1:-1]

    if not steps:
        # Number of time steps from an educated guess about how far to move
        with torch.no_grad():
            steps = vel.square().sum(dim=-1).max().sqrt().floor().int().item() + 1

    mom = mom0 = metric.forward(vel)
    vel = vel / steps
    disp = -vel
    idisp = vel

    for i in range(1, abs(steps)):
        if fast:
            # push current momentum using current velocity
            # JA:
            #   the update of u_t is not exactly as described in the paper,
            #   but describing this might be a bit tricky. The approach here
            #   was the most stable one I could find - although it does lose
            #   some energy as < v_t, u_t> decreases over time steps.
            jac = jacobian(vel, bound=metric.bound, add_identity=False).neg_()
            jac.diagonal(0, -1, -2).add_(1)
            mom = batchmatvec(jac.transpose(-1, -2), mom)
            mom = backend.push(mom, vel, bound=metric.bound)
        else:
            # push initial momentum using full flow
            jac = batchinv(jacobian(idisp, bound=metric.bound, add_identity=True))
            mom = batchmatvec(jac.transpose(-1, -2), mom0)
            mom = backend.push(mom, idisp, bound=metric.bound)

        # Convolve with Greens function of L
        # v_t <- inv(L) u_t
        vel = metric.inverse(mom)
        vel = vel.div_(steps)
        if verbose:
            print(f'{0.5*steps*(vel*mom).sum().item()/spatial.numel():6g}',
                  end='\n' if not (i % 5) else ' ', flush=True)

        # psi <- psi o (id - v/T)
        # JA:
        #   I found that simply using `psi <- psi - (D psi) v/T`
        #   was not so stable.
        disp = compose(disp, -vel, bound=metric.bound)
        idisp = compose(vel, idisp, bound=metric.bound)

    if verbose:
        print('')
    return disp, idisp
