import numpy as np

def random(shape, offset=0.5, initmult=0.1):
    return (np.random.random(shape).astype("float32")-offset)*initmult