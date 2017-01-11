from unittest import TestCase
from teafacto.core.base import Block
from teafacto.blocks.basic import Linear


class TestBlock(TestCase):
    def test_subclasses(self):
        b = Linear(10,5)
        p = b.get_probe()
        print p
