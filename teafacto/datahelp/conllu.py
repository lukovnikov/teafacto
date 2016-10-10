import numpy as np
from IPython import embed
from teafacto.procutil import wordids2string


class CoNLLUDataSet(object):
    def __init__(self,
                 trainp="../../data/ud/en-ud-train.conllu",
                 testp="../../data/ud/en-ud-test.conllu",
                 devp="../../data/ud/en-ud-dev.conllu",
                 maskid=-1):
        self.traindata = CoNLLUData(trainp, maskid=maskid)
        self.testdata = CoNLLUData(testp, maskid=maskid, worddic=self.traindata.worddic, tworddic=self.traindata.tworddic, udposdic=self.traindata.udposdic, enposdic=self.traindata.enposdic)
        self.devdata = CoNLLUData(devp, maskid=maskid, worddic=self.testdata.worddic, tworddic=self.testdata.tworddic, udposdic=self.testdata.udposdic, enposdic=self.testdata.enposdic)
        self.worddic = self.devdata.worddic
        self.rworddic = {v: k for k, v in self.worddic.items()}
        self.tworddic = self.devdata.tworddic
        self.rtworddic = {v: k for k, v in self.tworddic.items()}
        self.udposdic = self.devdata.udposdic
        self.rudposdic = {v: k for k, v in self.udposdic.items()}
        self.enposdic = self.devdata.enposdic
        self.renposdic = {v: k for k, v in self.enposdic.items()}


class CoNLLUData(object):
    def __init__(self, p, maskid=-1, worddic=None, tworddic=None, udposdic=None, enposdic=None):
        self.sentencemat = None
        self.tsentencemat = None
        self.udpostagmat = None
        self.enpostagmat = None

        self.maskid = maskid

        self.worddic = {} if worddic is None else worddic
        self.tworddic = {} if tworddic is None else tworddic
        self.udposdic = {} if udposdic is None else udposdic
        self.enposdic = {} if enposdic is None else enposdic

        self.load(p)

    def load(self, p):
        words = set()
        twords = set()
        udptags = set()
        enptags = set()
        sentences = []
        tsentences = []
        udpostags = []
        enpostags = []
        sentence = []
        tsentence = []
        udpostag = []
        enpostag = []
        maxlen = 0
        for line in open(p):
            if line[0] == "#":
                continue
            if len(line) == 0 or (len(line) == 1 and line[0] == "\n"):
                maxlen = max(maxlen, len(sentence))
                sentences.append(sentence)
                tsentences.append(tsentence)
                udpostags.append(udpostag)
                enpostags.append(enpostag)
                sentence = []
                tsentence = []
                udpostag = []
                enpostag = []
                continue
            worddata = line[:-1].split("\t")
            sentence.append(worddata[1])
            tsentence.append(worddata[2])
            udpostag.append(worddata[3])
            enpostag.append(worddata[4])
            words.add(worddata[1])
            twords.add(worddata[2])
            udptags.add(worddata[3])
            enptags.add(worddata[4])
        # transform to matrices, build dictionaries
        def addtodic(xs, dic):
            nextid = max(dic.values()) + 1 if len(dic) > 0 else 0
            for x in xs:
                if x not in dic:
                    dic[x] = nextid
                    nextid += 1
        addtodic(words, self.worddic)
        addtodic(twords, self.tworddic)
        addtodic(udptags, self.udposdic)
        addtodic(enptags, self.enposdic)
        sentencemat = self.maskid * np.ones((len(sentences), maxlen), dtype="int32")
        tsentencemat = self.maskid * np.ones((len(sentences), maxlen), dtype="int32")
        udpostagmat = self.maskid * np.ones((len(sentences), maxlen), dtype="int32")
        enpostagmat = self.maskid * np.ones((len(sentences), maxlen), dtype="int32")
        for i in range(len(sentences)):
            sentencemat[i, :len(sentences[i])] = [self.worddic[x] for x in sentences[i]]
            tsentencemat[i, :len(tsentences[i])] = [self.tworddic[x] for x in tsentences[i]]
            udpostagmat[i, :len(udpostags[i])] = [self.udposdic[x] for x in udpostags[i]]
            enpostagmat[i, :len(enpostags[i])] = [self.enposdic[x] for x in enpostags[i]]
        self.sentencemat = sentencemat
        self.tsentencemat = tsentencemat
        self.udpostagmat = udpostagmat
        self.enpostagmat = enpostagmat


if __name__ == "__main__":
    datas = CoNLLUDataSet("../../data/ud/en-ud-train.conllu",
                          "../../data/ud/en-ud-test.conllu",
                          "../../data/ud/en-ud-dev.conllu")
    embed()
