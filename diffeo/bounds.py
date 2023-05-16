"""
Boundary conditions
===================

There is no common convention to name boundary conditions.
Here's the list all possible aliases.

=========   ===========   =======================   =======================
Fourier     SciPy         Metric                    Description
=========   ===========   =======================   =======================
dft         wrap          circular                  c  d | a b c d |  a  b
dct2        reflect       neumann                   b  a | a b c d |  d  c
dct1        mirror                                  c  b | a b c d |  c  b
dst2                      dirichlet                -b -a | a b c d | -d -c
dst1                                               -a  0 | a b c d |  0 -d

We further define a flow-specific "sliding" boundary condition, which
uses a combination of dct2 and dst2:
=============================   =============================
X component                     Y component
=============================   =============================
 -f -e   -e -f -g -h   -h -g     -f -e    e  f  g  h   -h -g
 -b -a   -a -b -c -d   -d -c     -b -a    a  b  c  d   -d -c
-----------------------------   -----------------------------
  b  a |  a  b  c  d |  d  c     -b -a |  a  b  c  d | -d -c
  f  e |  e  f  g  h |  h  g     -f -e |  e  f  g  h | -h -g
  j  i |  i  j  k  l |  l  k     -j -i |  i  j  k  l | -l -k
  l  m |  m  n  o  p |  p  o     -n -m |  m  n  o  p | -p -o
-----------------------------   -----------------------------
 -n -m   -m -n -o -p   -p -o     -n -m    m  n  o  p   -p -o
 -j -i   -i -j -k -l   -l -k     -j -i    i  j  k  l   -l -k
"""
import torch

bound2dft = {
    # circulant
    'circulant': 'dft',
    'circ': 'dft',
    'c': 'dft',
    # neumann
    'neumann': 'dct2',
    'n': 'dct2',
    # dirichlet
    'dirichlet': 'dst2',
    'd': 'dst2',
    # scipy
    'reflect': 'dct2',
    'mirror': 'dct1',
    'wrap': 'dft',
    # zero
    'zeros': 'zero',
    'constant': 'zero',
}

dft2bound = {
    'dft': 'circulant',
    'dct': 'neumann',
    'dct2': 'neumann',
    'dst': 'dirichlet',
    'dst2': 'dirichlet',
}


def has_sliding(bound):
    return any(map(lambda x: x.lower()[0] == 's', bound))


def sliding2dft(bound, d):
    new_bound = []
    for i, b in enumerate(bound):
        if b[0].lower() == 's':
            # sliding
            new_bound.append('dst2' if i == d else 'dct2')
        else:
            new_bound.append(b)
    return new_bound


def dft(i, n):
    """Apply DFT (circulant/wrap) boundary conditions to an index

    Parameters
    ----------
    i : [tensor of] int
        Index
    n : int
        Length of the field of view

    Returns
    -------
    i : [tensor of] int
        Index that falls inside the field of view [0, n-1]
    s : [tensor of] {1, -1}
        Sign of the transformation (always 1 for dft)

    """
    if isinstance(i, int):
        return i % n, 1
    return i.remainder(n), 1


def dft_(i, n):
    """Apply DFT (circulant/wrap) boundary conditions to an index, in-place

    Parameters
    ----------
    i : [tensor of] int
        Index
    n : int
        Length of the field of view

    Returns
    -------
    i : [tensor of] int
        Index that falls inside the field of view [0, n-1]
    s : [tensor of] {1, -1}
        Sign of the transformation (always 1 for dft)

    """
    if isinstance(i, int):
        return dft(i, n)
    return i.remainder_(n), 1


def replicate(i, n):
    """Apply replicate (nearest/border) boundary conditions to an index

    Parameters
    ----------
    i : [tensor of] int
        Index
    n : int
        Length of the field of view

    Returns
    -------
    i : [tensor of] int
        Index that falls inside the field of view [0, n-1]
    s : [tensor of] {1, -1}
        Sign of the transformation (always 1 for replicate)

    """
    if isinstance(i, int):
        return min(max(i, 0), n-1), 1
    return i.clamp(min=0, max=n-1), 1


def replicate_(i, n):
    """Apply replicate (nearest/border) boundary conditions to an index, in-place

    Parameters
    ----------
    i : [tensor of] int
        Index
    n : int
        Length of the field of view

    Returns
    -------
    i : [tensor of] int
        Index that falls inside the field of view [0, n-1]
    s : [tensor of] {1, -1}
        Sign of the transformation (always 1 for replicate)

    """
    if isinstance(i, int):
        return replicate(i, n)
    return i.clamp_(min=0, max=n-1), 1


def dct2(i, n):
    """Apply DCT-II (reflect) boundary conditions to an index

    Parameters
    ----------
    i : [tensor of] int
        Index
    n : int
        Length of the field of view

    Returns
    -------
    i : [tensor of] int
        Index that falls inside the field of view [0, n-1]
    s : [tensor of] {1, -1}
        Sign of the transformation (always 1 for dct2)

    """
    n2 = n * 2
    if isinstance(i, int):
        i = n2 - 1 - ((-i-1) % n2) if i < 0 else i % n2
        i = n2 - 1 - i if i >= n else i
        return i, 1
    i = torch.where(i < 0, (n2-1) - (-i-1).remainder(n2),
                    i.remainder(n2))
    i = torch.where(i >= n, (n2 - 1) - i, i)
    return i, 1


def dct2_(i, n):
    """Apply DCT-II (reflect) boundary conditions to an index, in-place

    Parameters
    ----------
    i : [tensor of] int
        Index
    n : int
        Length of the field of view

    Returns
    -------
    i : [tensor of] int
        Index that falls inside the field of view [0, n-1]
    s : [tensor of] {1, -1}
        Sign of the transformation (always 1 for dct2)

    """
    if isinstance(i, int):
        return dct2(i, n)
    n2 = n*2
    pre = (i < 0)
    i[pre] = n2 - 1 - ((-i[pre]-1) % n2)
    i[~pre] = (i[~pre] % n2)
    post = (i >= n)
    i[post] = n2 - i[post] - 1
    return i, 1


def dct1(i, n):
    """Apply DCT-I (mirror) boundary conditions to an index

    Parameters
    ----------
    i : [tensor of] int
        Index
    n : int
        Length of the field of view

    Returns
    -------
    i : [tensor of] int
        Index that falls inside the field of view [0, n-1]
    s : [tensor of] {1, -1}
        Sign of the transformation (always 1 for dct)

    """
    if n == 1:
        if isinstance(i, int):
            return 0, 1
        return torch.zeros_like(i), 1
    else:
        n2 = (n - 1) * 2
        if isinstance(i, int):
            i = abs(i) % n2
            i = n2 - i if i >= n else i
            return i, 1
        i = i.abs().remainder(n2)
        i = torch.where(i >= n, -i + n2, i)
        return i, 1


def dct1_(i, n):
    """Apply DCT-I (mirror) boundary conditions to an index

    Parameters
    ----------
    i : [tensor of] int
        Index
    n : int
        Length of the field of view

    Returns
    -------
    i : [tensor of] int
        Index that falls inside the field of view [0, n-1]
    s : [tensor of] {1, -1}
        Sign of the transformation (always 1 for dct1)

    """
    if isinstance(i, int):
        return dct1(i, n)
    if n == 1:
        return i.zero_(), 1
    else:
        n2 = (n - 1) * 2
        i = i.abs_().remainder_(n2)
        mask = i >= n
        i[mask] *= -1
        i[mask] += n2
        return i, 1


def dst1(i, n):
    """Apply DST-I (antimirror) boundary conditions to an index

    Parameters
    ----------
    i : [tensor of] int
        Index
    n : int
        Length of the field of view

    Returns
    -------
    i : [tensor of] int
        Index that falls inside the field of view [0, n-1]
    s : [tensor of] {1, -1}
        Sign of the transformation

    """
    n2 = 2 * (n + 1)
    if isinstance(i, int):
        # sign
        ii = n - 1 - i if i < 0 else i
        ii = ii % n2
        x = 0 if ii == 0 else 1
        x = 0 if ii % (n + 1) == n else x
        x = -x if (ii / (n + 1)) % 2 > 0 else x
        # index
        i = -i - 2 if i < 0 else i
        i = i % n2
        i = n2 - 2 - i if i > n else i
        i = min(max(i, 0), n-1)
        return i, x

    one = torch.ones([1], dtype=torch.int8, device=i.device)
    zero = torch.zeros([1], dtype=torch.int8, device=i.device)
    first = torch.full([1], 0, dtype=i.dtype, device=i.device)
    last = torch.full([1], n - 1, dtype=i.dtype, device=i.device)

    i = torch.where(i < 0, -i - 2, i)
    i = i.remainder(n2)

    # sign
    x = torch.where(i.remainder(n + 1) == n, zero, one)
    x = torch.where((i / (n + 1)).remainder(2) > 0, -x, x)

    # index
    i = torch.where(i > n, -i + (n2 - 2), i)
    i = torch.where(i == -1, first, i)
    i = torch.where(i == n, last, i)
    return i, x


def dst1_(i, n):
    """Apply DST-I (antimirror) boundary conditions to an index, in-place

    Parameters
    ----------
    i : [tensor of] int
        Index
    n : int
        Length of the field of view

    Returns
    -------
    i : [tensor of] int
        Index that falls inside the field of view [0, n-1]
    s : [tensor of] {1, -1}
        Sign of the transformation

    """
    if isinstance(i, int):
        return dst1(i, n)

    one = torch.ones([1], dtype=torch.int8, device=i.device)
    zero = torch.zeros([1], dtype=torch.int8, device=i.device)

    n2 = 2 * (n + 1)

    mask = i < 0
    i[mask] += 2
    i[mask] *= -1
    i = i.remainder_(n2)

    # sign
    x = torch.where(i.remainder(n + 1) == n, zero, one)
    mask = (i / (n + 1)).remainder(2) > 0
    x *= 1 - 2 * mask

    # index
    mask = i > n
    i[mask] *= -1
    i[mask] += n2 - 2
    i.clamp_(0, n-1)
    return i, x


def dst2(i, n):
    """Apply DST-II (antireflect) boundary conditions to an index

    Parameters
    ----------
    i : [tensor of] int
        Index
    n : int
        Length of the field of view

    Returns
    -------
    i : [tensor of] int
        Index that falls inside the field of view [0, n-1]
    s : [tensor of] {1, -1}
        Sign of the transformation

    """
    if n == 1:
        return dct2(i, n)
    else:
        ii = torch.where(i < 0, n - 1 - i, i)
        x = torch.ones([1], dtype=torch.int8, device=i.device)
        x = torch.where((ii / n).remainder(2) > 0, -x, x)
        return dct2(i, n)[0], x


def dst2_(i, n):
    """Apply DST-II (antireflect) boundary conditions to an index, in-place

    Parameters
    ----------
    i : [tensor of] int
        Index
    n : int
        Length of the field of view

    Returns
    -------
    i : [tensor of] int
        Index that falls inside the field of view [0, n-1]
    s : [tensor of] {1, -1}
        Sign of the transformation

    """
    if n == 1:
        return dct2_(i, n)
    else:
        ii = torch.where(i < 0, n - 1 - i, i)
        x = torch.ones([1], dtype=torch.int8, device=i.device)
        x = torch.where((ii / n).remainder(2) > 0, -x, x)
        return dct2_(i, n)[0], x


nearest = border = replicate
reflect = neumann = dct2
mirror = dct1
antireflect = dirichlet = dst2
antimirror = dst1
wrap = circular = dft
