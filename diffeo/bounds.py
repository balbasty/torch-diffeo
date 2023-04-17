bound2dft = {
    'circulant': 'dft',
    'circ': 'dft',
    'c': 'dft',
    'neumann': 'dct2',
    'n': 'dct2',
    'dirichlet': 'dst2',
    'd': 'dst2',
}

dft2bound = {
    'dft': 'circulant',
    'dct': 'neumann',
    'dct2': 'neumann',
    'dst': 'dirichlet',
    'dst2': 'dirichlet',
}