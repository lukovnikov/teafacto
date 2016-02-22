import theano
from theano import tensor as T
from teafacto.blocks.datafeed import DataFeed
from teafacto.blocks.trainer import ModelTrainer, ContrastModelTrainer
from teafacto.blocks.core import Input, Block

# DO NOT USE THIS, USE BLOCKS INSTEAD


class Model(Block):
    '''
    A model contains the expression to optimize
    Provides fluent interface for training (configuring options like objective, optimizer, datafeed)
    Does not contain (1) objective or (2) optimizer or (3) datafeeder
    '''
    # dummy implementations, do override --------------------------
    def __init__(self, **kw):
        super(Model, self).__init__(**kw)
        self._predictf = None
        self.inputs = None
        self.output = None

    def initparams(self):
        ''' init params here, use block ways '''

    def initinputs(self):
        '''RETURNS INPUT BLOCK VARS'''
        return [] # list of inputs

    def apply(self, *inps): # theano vars as input, theano vars as output
        '''RETURNS BLOCK OUTPUT VAR'''
        return inps[0] # defines the operations to get to the output

    # may override: -------------------------------------------------
    def predict(self, inputdata):
        if self._predictf is None:
            self.build()
            self._predictf = theano.function(outputs=self.output.d, inputs=[x.d for x in self.inputs])
        return self._predictf(dict(zip([x.d for x in self.inputs], inputdata)))

    # do not override ------------------------------------------------
    def build(self): # stores block inputs and block output
        self.inputs = self.initinputs()
        self.output = self.apply(self.inputs)

    def train(self, inputdata, gold):
        # wrap data in datafeeds, generate gold var
        if not isinstance(inputdata, DataFeed):
            inputdata = DataFeed(inputdata)
        if not isinstance(gold, DataFeed):
            gold = DataFeed(gold)
        goldvar = Input(gold.ndim, gold.dtype, name="gold")
        trainer = self.gettrainer(goldvar.d)
        trainer.traindata = inputdata
        trainer.traingold = gold
        return trainer

    def gettrainer(self, goldvar):
        return ModelTrainer(self, goldvar)


class ContrastModel(Model): # the inner model defined should focus on the prediction of the score for one example
    def gettrainer(self, goldvar):
        return ContrastModelTrainer(self, goldvar)