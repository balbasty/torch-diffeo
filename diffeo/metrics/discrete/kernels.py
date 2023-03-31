import torch
from diffeo.utils import make_vector


def absolute(dim, voxel_size=1, dtype=None, device=None):
    """Precision matrix for the Absolute energy, as a convolution kernel.

    Parameters
    ----------
    dim : int
    voxel_size : float or sequence[float], default=1

    Returns
    -------
    kernel : (1,)*dim sparse tensor

    """
    dtype = dtype or torch.get_default_dtype()
    voxel_size = make_vector(voxel_size, dim, dtype=dtype, device=device)

    kernel = torch.sparse_coo_tensor(
        torch.zeros([dim, 1], dtype=torch.long, device=device),
        torch.ones([1], dtype=dtype, device=device),
        [1] * dim)

    kernel = torch.stack([kernel * vx.square() for vx in voxel_size], dim=0)
    return kernel


def membrane(dim, voxel_size=1, dtype=None, device=None):
    """Precision matrix for the Membrane energy, as a convolution kernel.

    Parameters
    ----------
    dim : int
    voxel_size : float or sequence[float], default=1

    Returns
    -------
    kernel : (dim, [3,]*dim) sparse tensor

    """
    dtype = dtype or torch.get_default_dtype()
    voxel_size = make_vector(voxel_size, dim, dtype=dtype, device=device)
    vx = voxel_size.square().reciprocal()

    # build sparse kernel
    kernel = [2 * vx.sum()]
    center_index = [1] * dim
    indices = [list(center_index)]
    for d in range(dim):
        # cross
        kernel += [-vx[d]] * 2
        index = list(center_index)
        index[d] = 0
        indices.append(index)
        index = list(center_index)
        index[d] = 2
        indices.append(index)
    indices = torch.as_tensor(indices, dtype=torch.long, device=vx.device)
    kernel = torch.as_tensor(kernel, dtype=dtype, device=device)
    kernel = torch.sparse_coo_tensor(indices.t(), kernel, [3] * dim)

    kernel = torch.stack([kernel * vx.square() for vx in voxel_size], dim=0)
    return kernel


def bending(dim, voxel_size=1, dtype=None, device=None):
    """Precision matrix for the Bending energy, as a convolution kernel.

    Parameters
    ----------
    dim : int
    voxel_size : float or sequence[float], default=1

    Returns
    -------
    kernel : (dim, dim, [5,]*dim) sparse tensor

    """
    dtype = dtype or torch.get_default_dtype()
    voxel_size = make_vector(voxel_size, dim, dtype=dtype, device=device)
    vx = voxel_size.square().reciprocal()
    vx2 = vx.square()
    cvx = torch.combinations(vx, r=2).prod(dim=-1)

    # build sparse kernel
    kernel = [6 * vx2.sum() + 8 * cvx.sum()]
    center_index = [2] * dim
    indices = [list(center_index)]
    for d in range(dim):
        # cross 1st order
        kernel += [-4*vx[d]*vx.sum()] * 2
        index = list(center_index)
        index[d] = 1
        indices.append(index)
        index = list(center_index)
        index[d] = 3
        indices.append(index)
        # cross 2nd order
        kernel += [vx2[d]] * 2
        index = list(center_index)
        index[d] = 0
        indices.append(index)
        index = list(center_index)
        index[d] = 4
        indices.append(index)
        for dd in range(d+1, dim):
            # off
            kernel += [2 * vx[d] * vx[dd]] * 4
            index = list(center_index)
            index[d] = 1
            index[dd] = 1
            indices.append(index)
            index = list(center_index)
            index[d] = 1
            index[dd] = 3
            indices.append(index)
            index = list(center_index)
            index[d] = 3
            index[dd] = 1
            indices.append(index)
            index = list(center_index)
            index[d] = 3
            index[dd] = 3
            indices.append(index)
    indices = torch.as_tensor(indices, dtype=torch.long, device=vx.device)
    kernel = torch.as_tensor(kernel, dtype=dtype, device=device)
    kernel = torch.sparse_coo_tensor(indices.t(), kernel, [5] * dim)

    kernel = torch.stack([kernel * vx.square() for vx in voxel_size], dim=0)
    return kernel


def lame_shear(dim, voxel_size=1, dtype=None, device=None):
    """Precision matrix for the Linear Elastic energy, as a convolution kernel.

    Parameters
    ----------
    dim : int
    voxel_size : float or sequence[float], default=1

    Returns
    -------
    kernel : (dim, dim, [3,]*dim) sparse tensor

    """
    dtype = dtype or torch.get_default_dtype()
    voxel_size = make_vector(voxel_size, dim, dtype=dtype, device=device)
    vx = voxel_size.square().reciprocal()

    # build sparse kernel
    kernel = []
    center_index = [1] * dim
    indices = []
    for d in range(dim):  # input channel
        kernel += [2 + 2*vx.sum()/vx[d]]
        index = [d, d, *center_index]
        indices.append(index)
        for dd in range(dim):  # cross
            if dd == d:
                kernel += [-2] * 2
            else:
                kernel += [-vx[dd]/vx[d]] * 2
            index = [d, d, *center_index]
            index[2 + dd] = 0
            indices.append(index)
            index = [d, d, *center_index]
            index[2 + dd] = 2
            indices.append(index)
        for dd in range(d+1, dim):  # output channel
            kernel += [-0.25] * 4
            index = [d, dd, *center_index]
            index[2 + d] = 0
            index[2 + dd] = 0
            indices.append(index)
            index = [dd, d, *center_index]
            index[2 + d] = 0
            index[2 + dd] = 0
            indices.append(index)
            index = [d, dd, *center_index]
            index[2 + d] = 2
            index[2 + dd] = 2
            indices.append(index)
            index = [dd, d, *center_index]
            index[2 + d] = 2
            index[2 + dd] = 2
            indices.append(index)
            kernel += [0.25] * 4
            index = [d, dd, *center_index]
            index[2 + d] = 0
            index[2 + dd] = 2
            indices.append(index)
            index = [dd, d, *center_index]
            index[2 + d] = 0
            index[2 + dd] = 2
            indices.append(index)
            index = [d, dd, *center_index]
            index[2 + d] = 2
            index[2 + dd] = 0
            indices.append(index)
            index = [dd, d, *center_index]
            index[2 + d] = 2
            index[2 + dd] = 0
            indices.append(index)

    indices = torch.as_tensor(indices, dtype=torch.long, device=vx.device)
    kernel = torch.as_tensor(kernel, dtype=dtype, device=device)
    kernel = torch.sparse_coo_tensor(indices.t(), kernel, [dim, dim] + [3] * dim)

    return kernel


def lame_div(dim, voxel_size=1, dtype=None, device=None):
    """Precision matrix for the Linear Elastic energy, as a convolution kernel.

    Parameters
    ----------
    dim : int
    voxel_size : float or sequence[float], default=1 (actually unused)

    Returns
    -------
    kernel : (dim, dim, [3,]*dim) sparse tensor

    """
    dtype = dtype or torch.get_default_dtype()

    # build sparse kernel
    kernel = []
    center_index = [1] * dim
    indices = []
    for d in range(dim):  # input channel
        kernel += [2]
        index = [d, d, *center_index]
        indices.append(index)
        kernel += [-1] * 2
        index = [d, d, *center_index]
        index[2 + d] = 0
        indices.append(index)
        index = [d, d, *center_index]
        index[2 + d] = 2
        indices.append(index)
        for dd in range(d+1, dim):  # output channel
            for d1 in range(dim):   # interation 1
                for d2 in range(d + 1, dim):  # interation 2
                    kernel += [-0.25] * 4
                    index = [d, dd, *center_index]
                    index[2 + d1] = 0
                    index[2 + d2] = 0
                    indices.append(index)
                    index = [dd, d, *center_index]
                    index[2 + d1] = 0
                    index[2 + d2] = 0
                    indices.append(index)
                    index = [d, dd, *center_index]
                    index[2 + d1] = 2
                    index[2 + d2] = 2
                    indices.append(index)
                    index = [dd, d, *center_index]
                    index[2 + d1] = 2
                    index[2 + d2] = 2
                    indices.append(index)
                    kernel += [0.25] * 4
                    index = [d, dd, *center_index]
                    index[2 + d1] = 0
                    index[2 + d2] = 2
                    indices.append(index)
                    index = [dd, d, *center_index]
                    index[2 + d1] = 0
                    index[2 + d2] = 2
                    indices.append(index)
                    index = [d, dd, *center_index]
                    index[2 + d1] = 2
                    index[2 + d2] = 0
                    indices.append(index)
                    index = [dd, d, *center_index]
                    index[2 + d1] = 2
                    index[2 + d2] = 0
                    indices.append(index)

    indices = torch.as_tensor(indices, dtype=torch.long, device=device)
    kernel = torch.as_tensor(kernel, dtype=dtype, device=device)
    kernel = torch.sparse_coo_tensor(indices.t(), kernel, [dim, dim] + [3] * dim)

    return kernel
