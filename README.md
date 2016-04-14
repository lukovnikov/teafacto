# TeaFacto

[![Build Status](https://travis-ci.org/lukovnikov/teafacto.svg?branch=master)](https://travis-ci.org/lukovnikov/teafacto)
[![Coverage Status](https://coveralls.io/repos/github/lukovnikov/teafacto/badge.svg?branch=master)](https://coveralls.io/github/lukovnikov/teafacto?branch=master)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/lukovnikov/teafacto/blob/master/LICENSE)

:tea::factory:

## What is this?
TeaFacto tries to provide a simple and flexible way to define and train (neural network) machine learning models.

## How does it work?
Currently, the code relies on Theano as the computational backend.

Instead of providing a layer-based abstraction like Keras or Lasagne, TeaFacto follows a block-based
(like Block Bricks) paradigm to provide more flexibility for the definition of complex models.

## What is a block in TeaFacto?
A block is a (optionally) parameterized operation on a set of inputs.

TeaFacto wraps Theano operations as blocks, which allows to use both blocks and normal Theano code in the same place.

Blocks may also have parameters, which are automatically handled by TeaFacto.
The only thing you need to do is to declare them and use them.

A simple block can be defined as follows:
```python
from teafacto.core.base import tensorops as T
from teafacto.core.base import Block, param
from teafacto.blocks.basic import Softmax

class Dummy(Block):
    def __init__(self, indim=1000, dim=50, outdim=10, **kw):
        super(Dummy, self).__init__(**kw)
        self.W = param((indim, dim), name="embedder").uniform()
        self.O = param((dim, outdim), name="output").uniform()

    def apply(self, idxs):
        return Softmax()(T.dot(self.W[idxs, :], self.O))
```
Parameters are declared by using the ```param()``` function with a initializer method (```.uniform()```),
which produces a ```Parameter``` object, or by using ```Parameter```'s constructor explicitly.

All blocks have an ```apply()``` method where the symbolic operations on its inputs are defined.

## How do I define a model?
There are no dedicated models in TeaFacto.
Every block can be trained without wrapping it in a model object.
So to declare a model, you simply need to define a new block.

## How do I train a block?
Like this:
```python
import numpy as np
from teafacto.util import argprun

def run(
        epochs=1,
        dim=10,
        vocabsize=2000,
        innerdim=100,
        lr=0.02,
        numbats=100
    ):

    data = np.arange(0, vocabsize)
    labels = np.random.randint(0, dim, (vocabsize,))
    block = Dummy(indim=vocabsize, dim=innerdim, outdim=dim)

    block.train([data], labels).adagrad(lr=lr).neg_log_prob()\
         .split_validate(5, random=True).neg_log_prob().accuracy().autosave\
         .train(numbats=numbats, epochs=epochs)

if __name__ == "__main__":
    argprun(run)
```
Calling the ```.train()``` method on a block and passing in the training data and the labels returns a
trainer that provides a fluent interface for training settings.
Under the hood, the trainer automatically creates input variables based on the provided input data.

The same is done for prediction:
```python
data = np.arange(0, vocabsize)
predicted_labels = block.predict(data)
```
In fact, you probably won't need to create any variables at all.