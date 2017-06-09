from unittest import TestCase
from teafacto.blocks.seq.neurex import DGTN
import numpy as np
from teafacto.core.base import Val


class TestDGTN_Actions(TestCase):
    def setUp(self):
        reltensor = np.random.random((2,4,4))
        reltensor = np.asarray([
            [
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [1, 0, 1, 0],
                [0, 1, 0, 0],
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]
            ]
        ])
        dgtn = DGTN(reltensor, nsteps=5, entembdim=10, relembdim=10, actembdim=10)
        self.dgtn = dgtn
        self.mainp = np.asarray([[0,0,0,1], [1,1,1,1], [0,1,0,0]])
        self.auxp = np.asarray([[1,0,0,1], [1,1,0,0], [0,0,0,0]])

    def test_exec_preserve(self):
        mainp, auxp = self.mainp, self.auxp
        newmain, newaux = self.dgtn._exec_preserve(Val(mainp), Val(auxp))
        self.assertTrue(np.allclose(mainp, newmain.d.eval()))
        self.assertTrue(np.allclose(auxp, newaux.d.eval()))

    def test_exec_find(self):
        w = np.random.random((3, 4))
        newmain, newaux = self.dgtn._exec_find(Val(w), Val(self.mainp), Val(self.auxp))
        self.assertTrue(np.allclose(w, newmain.d.eval()))
        self.assertTrue(np.allclose(self.mainp, newaux.d.eval()))

    def test_exec_hop(self):
        w = np.asarray([[1,0], [1,0], [0.2, 0.8]])
        newmain, newaux = self.dgtn._exec_hop(Val(w), Val(self.mainp), Val(self.auxp))
        #print self.mainp
        #print newmain.eval()
        self.assertTrue(np.allclose(newmain.eval(), np.asarray([[ 0., 1., 0., 0.], [ 1.,   1.,   1.,   0., ], [ 0.,  0.,   0.2,  0. ]])))
        self.assertTrue(np.allclose(newaux.d.eval(), self.auxp))

    def test_exec_intersect(self):
        newmain, newaux = self.dgtn._exec_intersect(Val(self.mainp), Val(self.auxp))
        print newmain.eval()
        self.assertTrue(np.allclose(newmain.eval(), np.asarray([[0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 0, 0]])))
        self.assertTrue(np.allclose(newaux.eval(), np.zeros_like(newaux.eval())))

    def test_exec_union(self):
        newmain, newaux = self.dgtn._exec_union(Val(self.mainp), Val(self.auxp))
        print newmain.eval()
        self.assertTrue(np.allclose(newmain.eval(), np.asarray([[1, 0, 0, 1], [1, 1, 1, 1], [0, 1, 0, 0]])))
        self.assertTrue(np.allclose(newaux.eval(), np.zeros_like(newaux.eval())))

    def test_exec_difference(self):
        newmain, newaux = self.dgtn._exec_difference(Val(self.mainp), Val(self.auxp))
        print newmain.eval()
        self.assertTrue(np.allclose(newmain.eval(), np.asarray([[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 0]])))
        self.assertTrue(np.allclose(newaux.eval(), np.zeros_like(newaux.eval())))

    def test_exec_swap(self):
        newmain, newaux = self.dgtn._exec_swap(Val(self.mainp), Val(self.auxp))
        self.assertTrue(np.allclose(newmain.d.eval(), self.auxp))
        self.assertTrue(np.allclose(newaux.d.eval(), self.mainp))

    # tests for auxiliary methods
    def test_get_att(self):
        crit = np.asarray([[0,1,1,0], [0,1,0,0]])
        data = np.asarray([[0,0,0,0],
                           [0,1,0,0],
                           [0,0,1,0],
                           [0,1,1,0]])
        attw = self.dgtn._get_att(Val(crit), Val(data))
        print attw.eval()
        self.assertTrue(np.allclose(np.sum(attw.eval(), axis=1), np.ones((2,))))

    def test_summarize_by_prob(self):
        w = np.random.random((2, 4))
        data = np.random.random((4, 6))
        res = self.dgtn._summarize_by_prob(w, data)
        self.assertEqual(res.eval().shape, (2, 6))

    def test_summarize_by_pointer(self):
        w = np.random.random((2, 4))
        data = np.random.random((4, 6))
        res = self.dgtn._summarize_by_pointer(w, data)
        self.assertEqual(res.eval().shape, (2, 6))




