from teafacto.core.base import tensorops as T, Val, param
import numpy as np

x = Val(np.random.random((10,10)))
#y = Val(np.random.random((10,10)))
y = param((10, 10), name="y").uniform()
w = param((10, 10), name="w").uniform()

#z = T.dot(x, y)
z = (x + y)
u = z * w
print z.allparams
print T.dot
print T.nnet.conv.conv2d
print u.norm(2).allparams