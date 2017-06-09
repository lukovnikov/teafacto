from unittest import TestCase
from teafacto.blocks.seq.neurex import DGTN
import numpy as np
from teafacto.core.base import Val

from teafacto.blocks.basic import VectorEmbed
from teafacto.blocks.seq.encdec import EncDec


class TestDGTN_without_attention(TestCase):
    def test_output_pointer_shape(self):
        # encoder
        encvocsize = 22
        encembdim = 11
        inpenc = VectorEmbed(encvocsize, encembdim)
        # DGTN
        reltensor = np.asarray([
            [
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [1, 0, 1, 1],
                [0, 1, 0, 0],
            ],
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 0, 0]
            ]
        ])
        dgtn = DGTN(reltensor=reltensor, nsteps=5, entembdim=10, relembdim=9, actembdim=8)
        # internal decoder without attention
        dec = EncDec(encoder=inpenc,
                     inconcat=False, outconcat=False, stateconcat=True, concatdecinp=False,
                     updatefirst=True,
                     inpemb=None, indim=dgtn.get_indim(),
                     innerdim=33)
        dgtn.set_core(dec)

        batsize = 7
        data = np.random.randint(0, encvocsize, (batsize,))
        out = dgtn(Val(data))

        res = out.eval()
        self.assertEqual(res.shape, (batsize, 4))

    def test_output_pointer_shape_given_pointer(self):
        # encoder
        encvocsize = 22
        encembdim = 11
        inpenc = VectorEmbed(encvocsize, encembdim)
        # DGTN
        reltensor = np.asarray([
            [
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [1, 0, 1, 1],
                [0, 1, 0, 0],
            ],
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 0, 0]
            ]
        ])
        dgtn = DGTN(reltensor=reltensor, nsteps=5, entembdim=10, relembdim=9, actembdim=8)
        # internal decoder without attention
        dec = EncDec(encoder=inpenc,
                     inconcat=False, outconcat=False, stateconcat=True, concatdecinp=False,
                     updatefirst=True,
                     inpemb=None, indim=dgtn.get_indim(),
                     innerdim=33)
        dgtn.set_core(dec)

        batsize = 7

        initptr = np.asarray([[1,0,0,0]]*batsize).astype("float32")
        print initptr
        data = np.random.randint(0, encvocsize, (batsize,))
        out = dgtn(Val(data), Val(initptr))

        res = out.eval()
        self.assertEqual(res.shape, (batsize, 4))


class TestDGTN_Actions_and_Helpers(TestCase):
    def setUp(self):
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

    def test_merge_exec(self):
        newps = ([[0,1,0,0], [0,0,1,0]], [[1,0,0,0],[0,0,0,1]])
        oldmain = [[0,0,0,0],[0,0,0,0]]
        oldaux = [[0,0,0,0],[0,0,0,0]]
        w = [0.5, 1]
        newmain, newaux = self.dgtn._merge_exec((Val(newps[0]), Val(newps[1])), Val(oldmain), Val(oldaux), Val(w))
        self.assertTrue(np.allclose(newmain.eval(),
                                    np.asarray([[0, 0.5, 0, 0], [0,0,1,0]])))
        print newmain.eval()
        print newaux.eval()

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

    def test_get_indim(self):
        indim = self.dgtn.get_indim()
        self.assertEqual(indim, 50)






