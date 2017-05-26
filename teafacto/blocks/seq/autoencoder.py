# autoencoders for sequences

from teafacto.util import argprun, ticktock
from teafacto.blocks.seq.rnn import SeqEncoder, SeqDecoder
from teafacto.blocks.basic import Dropout
from teafacto.core.base import Block, tensorops as T


class SeqAutoEncoder(Block):
    def __init__(self, encoder, decoder, dropout=None, **kw):
        super(SeqAutoEncoder, self).__init__(**kw)
        self.encoder = encoder
        self.decoder = decoder
        self.dropout = Dropout(dropout)

    def apply(self, x):
        encoding = self.encoder(x)
        encoding = self.dropout(encoding)
        decoding = self.decoder(encoding, x[:, :-1])
        return decoding


class SeqEncoderGroup(Block):
    def __init__(self, encoders, mode="concat", **kw):
        super(SeqEncoderGroup, self).__init__(**kw)
        self.encoders = encoders
        self.mode = mode

    def apply(self, *x):
        encodings = []
        for xe, encoder in zip(x, self.encoders):
            enco = encoder(xe)
            encodings.append(enco)
        if self.mode == "concat":
            encoding = T.concatenate(encodings, axis=-1)
        elif self.mode == "sum":
            encoding = reduce(lambda x, y: x + y, encodings, 0)
        else:
            raise Exception("unrecognized mode")
        return encoding


class MultiSeqAutoEncoder(Block):
    def __init__(self, encoders, decoders, dropout=None, mode="concat", **kw):
        super(MultiSeqAutoEncoder, self).__init__(**kw)
        self.encoders = encoders
        self.decoders = decoders
        self.dropout = Dropout(dropout)
        self.mode = mode

    def get_encoder(self):
        return SeqEncoderGroup(self.encoders, self.mode)

    def apply(self, *x):
        encodings = []
        for xe, encoder in zip(x, self.encoders):
            encoding = encoder(xe)
            encodings.append(encoding)
        if self.mode == "concat":
            decodings = []
            for encoding, decoder, xe in zip(encodings, self.decoders, x):
                encoding = self.dropout(encoding)
                decoding = decoder(encoding, xe[:, :-1])
                decodings.append(decoding)
        elif self.mode == "sum":
            encoding = reduce(lambda x, y: x + y, encodings, 0)
            encoding = self.dropout(encoding)
            decodings = [decoder(encoding, xe[:, :-1]) for decoder, xe in zip(self.decoders, x)]
        else:
            raise Exception("invalid mode {}".format(self.mode))
        return tuple(decodings)
