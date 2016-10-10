import teafacto as F
from teafacto.blocks.basic import Linear

x = F.input(1, "float32")
l = Linear(10, 10)
y = l(x)
l2 = Linear(10, 10)
y = l2(y)

print y.allparams