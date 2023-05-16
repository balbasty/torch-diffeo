import torch
from diffeo.utils import cartesian_grid, ensure_list
from diffeo.diffdiv import diff
from diffeo.backends import interpol as interpol_backend
from diffeo.linalg import batchmatvec, batchdot
from diffeo.bounds import bound2dft, sliding2dft, has_sliding


def add_identity_(flow):
    """Adds the identity grid to a displacement field, inplace.

    Parameters
    ----------
    flow : (..., *shape, dim) tensor
        Displacement field

    Returns
    -------
    flow : (..., *shape, dim) tensor
        Transformation field

    """
    dim = flow.shape[-1]
    spatial = flow.shape[-dim-1:-1]
    grid = cartesian_grid(spatial, dtype=flow.dtype, device=flow.device)
    flow = flow.movedim(-1, 0)
    for i, grid1 in enumerate(grid):
        flow[i].add_(grid1)
    flow = flow.movedim(0, -1)
    return flow


def sub_identity_(flow):
    """Subtracts the identity grid from a transformation field, inplace.

    Parameters
    ----------
    flow : (..., *shape, dim) tensor
        Transformation field

    Returns
    -------
    flow : (..., *shape, dim) tensor
        Displacement field

    """
    dim = flow.shape[-1]
    spatial = flow.shape[-dim-1:-1]
    grid = cartesian_grid(spatial, dtype=flow.dtype, device=flow.device)
    flow = flow.movedim(-1, 0)
    for i, grid1 in enumerate(grid):
        flow[i].sub_(grid1)
    flow = flow.movedim(0, -1)
    return flow


def add_identity(flow):
    """Adds the identity grid to a displacement field.

    Parameters
    ----------
    flow : (..., *shape, dim) tensor
        Displacement field

    Returns
    -------
    flow : (..., *shape, dim) tensor
        Transformation field

    """
    return add_identity_(flow.clone())


def sub_identity(flow):
    """Subtracts the identity grid from a transformation field.

    Parameters
    ----------
    flow : (..., *shape, dim) tensor
        Transformation field

    Returns
    -------
    flow : (..., *shape, dim) tensor
        Displacement field

    """
    return sub_identity_(flow.clone())


def identity(shape, **backend):
    """Returns an identity transformation field.

    Parameters
    ----------
    shape : (dim,) sequence of int
        Spatial dimension of the field.

    Returns
    -------
    grid : (*shape, dim) tensor
        Transformation field

    """
    backend.setdefault('dtype', torch.get_default_dtype())
    return torch.stack(cartesian_grid(shape, **backend), dim=-1)


def affine_field(affine, shape, add_identity=False):
    """Generate an affine flow field

    Parameters
    ----------
    affine : (..., D+1, D+1) tensor
        Affine matrix
    shape : (D,) list[int]
        Lattice size

    Returns
    -------
    flow : (..., *shape, D) tensor, Affine flow

    """
    ndim = len(shape)
    backend = dict(dtype=affine.dtype, device=affine.device)

    # add spatial dimensions so that we can use batch matmul
    for _ in range(ndim):
        affine = affine.unsqueeze(-3)
    lin, trl = affine[..., :ndim, :ndim], affine[..., :ndim, -1]

    # create affine transform
    flow = identity(shape, **backend)
    flow = lin.matmul(flow.unsqueeze(-1)).squeeze(-1)
    flow = flow.add_(trl)

    # subtract identity to get a flow
    if not add_identity:
        flow = sub_identity_(flow)

    return flow


def jacobian(flow, bound='circulant', voxel_size=1,
             has_identity=False, add_identity=None):
    """Compute the Jacobian of a transformation field

    Parameters
    ----------
    flow : (..., *spatial, dim) tensor
        Transformation or displacement field
    bound : [list of] {'circulant', 'neumann', 'dirichlet', 'sliding'}, default='circulant'
        Boundary condition
    voxel_size : [sequence of] float, default=1
        Voxel size
    has_identity : bool, default=False
        Whether the input is a transformation (True) or displacement
        (False) field.
    add_identity : bool, default=`has_identity`
        Adds the identity to the Jacobian of the displacement,
        making it the jacobian of the transformation.

    Returns
    -------
    jac : (..., *spatial, dim, dim) tensor
        Jacobian. In each matrix: jac[i, j] = d psi[i] / d xj

    """
    ndim = flow.shape[-1]
    if has_identity:
        flow = sub_identity(flow)
    bound = ensure_list(bound, ndim)
    bound = list(map(lambda x: bound2dft.get(x, x), bound))
    if has_sliding(bound):
        dims = list(range(-ndim, 0))
        jac = flow.new_zeros([*flow.shape, ndim])
        for d in range(ndim):
            bound1 = sliding2dft(bound, d)
            jac[..., d] = diff(flow[..., d], dim=dims, bound=bound1,
                               voxel_size=voxel_size, side='c')
    else:
        dims = list(range(-ndim-1, -1))
        jac = diff(flow, dim=dims, bound=bound,
                   voxel_size=voxel_size, side='c')
    if add_identity is None:
        add_identity = has_identity
    if add_identity:
        torch.diagonal(jac, 0, -1, -2).add_(1)
    return jac


def jacdet(flow, bound='circulant', voxel_size=1,
           has_identity=False, add_identity=True):
    """Compute the determinant of the Jacobian of a transformation field

    Parameters
    ----------
    flow : (..., *spatial, dim) tensor
        Transformation or displacement field
    bound : [list of] {'circulant', 'neumann', 'dirichlet', 'sliding'}, default='circulant'
        Boundary condition
    voxel_size : [sequence of] float, default=1
        Voxel size
    has_identity : bool, default=False
        Whether the input is a transformation (True) or displacement
        (False) field.
    add_identity : bool, default=`has_identity`
        Adds the identity to the Jacobian of the displacement,
        making it the jacobian of the transformation.

    Returns
    -------
    det : (..., *spatial) tensor
        Jacobian determinant.

    """
    jac = jacobian(
        flow,
        bound=bound,
        voxel_size=voxel_size,
        has_identity=has_identity,
        add_identity=add_identity,
    )
    return jac.det()


def compose(flow_left, flow_right, bound='circulant', has_identity=False, backend=interpol_backend):
    """Compute flow_left o flow_right

    Parameters
    ----------
    flow_left : (..., *shape, D) tensor
    flow_right : (..., *shape, D) tensor
    bound : [list of] {'circulant', 'neumann', 'dirichlet', 'sliding'}, default='circulant'
    has_identity : bool, default=False
    backend : module

    Returns
    -------
    flow : (..., *shape, D) tensor

    """
    if has_identity:
        flow_left = sub_identity(flow_left)
    flow = backend.pull(flow_left, flow_right, bound=bound, has_identity=has_identity)
    if flow.requires_grad:
        flow = flow + flow_right
    else:
        flow += flow_right
    return flow


def compose_jacobian(jac, rhs, lhs=None, bound='circulant', has_identity=False,
                     backend=interpol_backend):
    """Jacobian of the composition `(lhs)o(rhs)`

    Parameters
    ----------
    jac : (..., *spatial, ndim, ndim) tensor
        Jacobian of input RHS transformation
    rhs : (..., *spatial, ndim) tensor
        Right-hand side transformation
    lhs : (..., *spatial, ndim) tensor, default=`rhs`
        Left-hand side small displacement
    bound : [list of] {'circulant', 'neumann', 'dirichlet', 'sliding'}, default='circulant'
        Boundary condition
    has_identity : bool, default=False
        Whether the leht-hand side is a transformation (True) or
        displacement (False) field.

    Returns
    -------
    composed_jac : (..., *spatial, ndim, ndim) tensor
        Jacobian of composition

    """
    if lhs is None:
        lhs = rhs
    if has_identity:
        lhs = sub_identity(lhs)
    ndim = rhs.shape[-1]
    jac = jac.transpose(-1, -2)
    new_jac = torch.empty_like(jac)
    bound = ensure_list(bound, ndim)
    bound = list(map(lambda x: bound2dft.get(x, x), bound))
    # NOTE: loop across dimensions to save memory
    for d in range(ndim):
        bound1 = sliding2dft(bound, d)
        jac_left = diff(lhs[..., d], bound=bound1, dim=range(-ndim, 0))
        jac_left = backend.pull(jac_left, rhs, bound=bound1, has_identity=has_identity)
        jac_left[..., d] += 1
        new_jac[..., d] = batchmatvec(jac, jac_left)
    return new_jac.transpose(-1, -2)


def jacmatvec(flow, vec, bound='circulant', voxel_size=1,
              has_identity=False, transpose=False):
    r"""Compute the matrix-vector product between the Jacobian of a
    transformation field and a vector field.

    Parameters
    ----------
    flow : (..., *spatial, dim) tensor
        Transformation or displacement field
    vec : (..., *spatial, dim) tensor
        Vector field
    bound : [list of] {'circulant', 'neumann', 'dirichlet', 'sliding'}, default='circulant'
        Boundary condition
    voxel_size : [sequence of] float, default=1
        Voxel size
    has_identity : bool, default=False
        Whether the input is a transformation (True) or displacement
        (False) field.
    transpose : bool, default=False
        Transpose the Jacobian before taking the product.

        - If `False`: return $\nu_i = \sum_j \frac{\partial \phi_i}{\partial x_j} v_j$
        - If `True`: return $\nu_i = \sum_i \frac{\partial \phi_j}{\partial x_i} v_j$

    Returns
    -------
    matvec : (..., *spatial, dim) tensor
        Matrix-vector product

    """
    ndim = flow.shape[-1]
    if has_identity:
        flow = sub_identity(flow)
    bound = ensure_list(bound, ndim)
    bound = list(map(lambda x: bound2dft.get(x, x), bound))
    dims = list(range(-ndim, 0))
    fullshape = torch.broadcast_shapes(flow.shape, vec.shape)
    if transpose:
        matvec = flow.new_zeros(fullshape)
    else:
        matvec = flow.new_empty(fullshape)
    for d in range(ndim):
        bound1 = sliding2dft(bound, d)
        jac = diff(flow[..., d], dim=dims, bound=bound1,
                   voxel_size=voxel_size, side='c')
        jac[..., d] += 1
        if transpose:
            matvec.addcmul_(jac, vec[..., d, None])
        else:
            matvec[..., d] = batchdot(jac, vec)
    return matvec


def bracket(vel_left, vel_right, bound='circulant', has_identity=False,
            backend=interpol_backend):
    """Compute the Lie bracket of two SVFs

    Parameters
    ----------
    vel_left : (..., *shape, D) tensor
    vel_right : (..., *shape, D) tensor
    bound : [list of] {'circulant', 'neumann', 'dirichlet', 'sliding'}, default='circulant'
    has_identity : bool, default=False
    backend : module

    Returns
    -------
    bkt : (..., *shape, D) tensor

    """
    return (compose(vel_left, vel_right, bound, has_identity, backend) -
            compose(vel_right, vel_left, bound, has_identity, backend))
