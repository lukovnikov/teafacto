__author__ = 'denis'

from teafacto.core.base import Input
from teafacto.core.base import param, Val, Block, Parameter
from teafacto.blocks.objectives import *


def input(ndim, dtype="float32", name=None, **kw):
    return Input(ndim, dtype, name=name, **kw)


def input_from(nparray, name=None, **kw):
    return input(nparray.ndim, nparray.dtype, name=name, **kw)
