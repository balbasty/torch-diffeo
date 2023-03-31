from contextlib import contextmanager
from . import interpol


default_backend = interpol


@contextmanager
def backend(bck):
    saved_backend = default_backend
    try:
        globals()['default_backend'] = bck
        yield bck
    finally:
        globals()['default_backend'] = saved_backend

