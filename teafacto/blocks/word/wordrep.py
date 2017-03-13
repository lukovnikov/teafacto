from teafacto.core import Block, T, param
from teafacto.blocks.activations import Sigmoid, Tanh
from teafacto.blocks.basic import Forward


class WordEmbCharEncMerge(Block):
    """ embeds words and encodes characters and merges """
    def __init__(self, embedder, encoder, **kw):
        super(WordEmbCharEncMerge, self).__init__(**kw)
        self.embedder = embedder
        self.encoder = encoder      # must output the same dim as embedder
        self.outdim = self.embedder.outdim

    def apply(self, x):     # (batsize, seqlen, charlen + 1)
        wordids = x[:, :, 0]    # (batsize, seqlen)
        wordembs = self._get_word_embs(wordids)
        mask = wordembs.mask
        charseqs = x[:, :, 1:]  # (batsize, seqlen, charlen)
        charembs = self._get_char_embs(charseqs)
        embs = self._merge_word_char(wordembs, charembs)
        embs.mask = mask    # (batsize, seqlen, some_emb_dim)
        return embs

    def _get_word_embs(self, wordids):
        wordembs = self.embedder(wordids)   # (batsize, seqlen, embdim)
        return wordembs

    def _get_char_embs(self, charseqs):
        chars_shep = charseqs.shape[:-1]
        flat_charseqs = charseqs.reshape((-1, charseqs.shape[-1]))  # (batsize * seqlen, charlen)
        flat_charembs = self.encoder(flat_charseqs)  # (batsize * seqlen, embdim)
        charembs = flat_charembs.reshape((chars_shep[0], chars_shep[1], flat_charembs.shape[-1]))  # (batsize, seqlen, embdim)
        return charembs

    def _merge_word_char(self, wordembs, charembs):
        raise NotImplementedError("use subclass")


class WordEmbCharEncConcat(WordEmbCharEncMerge):
    def _merge_word_char(self, wordembs, charembs):
        ret = T.concatenate([wordembs, charembs], axis=-1)
        return ret


class WordEmbCharEncGate(WordEmbCharEncMerge):
    def __init__(self, embedder, encoder, gatedim=None, dropout=False, **kw):
        super(WordEmbCharEncGate, self).__init__(embedder, encoder, **kw)
        gatedim = self.outdim if gatedim is None else gatedim
        self.l1 = Forward(self.outdim * 2, gatedim, nobias=True, dropout=dropout)
        self.l2 = Forward(gatedim, self.outdim, activation=Sigmoid(), nobias=True, dropout=dropout)

    def _merge_word_char(self, wordembs, charembs):
        conc = T.concatenate([wordembs, charembs], axis=-1)
        gate = self.l1(conc)
        gate = self.l2(gate)    # (batsize, seqlen, embdim)
        word = gate * wordembs
        char = (1 - gate) * charembs
        ret = word + char
        return ret


class WordEmbCharEncCtxGate(WordEmbCharEncMerge):
    def __init__(self, embedder, encoder, gate_enc=None, **kw):
        super(WordEmbCharEncCtxGate, self).__init__(embedder, encoder, **kw)
        self.gate = gate_enc

    def _merge_word_char(self, wordembs, charembs):
        conc = T.concatenate([wordembs, charembs], axis=-1)
        gate = self.gate(conc)      # (batsize, seqlen, embdim)
        ret = gate * wordembs + (1 - gate) * charembs
        return ret



