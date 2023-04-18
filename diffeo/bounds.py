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
