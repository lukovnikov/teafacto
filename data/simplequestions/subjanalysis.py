from teafacto.scripts.simplequestions.subjdet import loadlabels
from teafacto.blocks.lang.wordembed import Glove
from teafacto.util import tokenize
import pickle


def run(labelp="labels.map", datap="datamat.word.pkl"):
    labeldic = loadlabels(labelp)
    glove = Glove(50)
    print "the" in glove
    len(labeldic)
    x = pickle.load(open(datap))
    print len(x["train"][0])
    # get entities without labels
    allentids = sorted(x["entdic"].items(), key=lambda (a, b): b)[:x["numents"]]
    print allentids[:20]
    entidsnotinlabeldic = set(map(lambda (a, b): a, allentids)).difference(set(labeldic.keys()))
    print len(entidsnotinlabeldic), list(entidsnotinlabeldic)[:20]
    wd = {v: k for k, v in x["worddic"].items()}
    ed = {v: k for k, v in x["entdic"].items()}
    alltrainents = set(x["train"][1][:, 0])
    allvalidents = set(x["valid"][1][:, 0])
    alltestents = set(x["test"][1][:, 0])
    print "%d/%d (%.2f%%) test set entities not in training " % (len(alltestents.difference(alltrainents)), len(alltestents), len(alltestents.difference(alltrainents))*100./len(alltestents))
    print "%d/%d (%.2f%%) validation set entities not in training " % (
        len(allvalidents.difference(alltrainents)), len(allvalidents),
        len(allvalidents.difference(alltrainents)) * 100. / len(allvalidents))
    #print len(allvalidents.difference(alltrainents))

    # gather all words in entity labels
    labelwords = {}
    labelwordsnotinglove = set()
    for label in labeldic.values():
        for labelw in tokenize(label):
            if labelw not in labelwords:
                labelwords[labelw] = 0
            labelwords[labelw] += 1
            if labelw not in glove:
                labelwordsnotinglove.add(labelw)

    print "%d unique words in labels" % len(labelwords)
    print "%d words not in glove" % len(labelwordsnotinglove)

    for split in ["train", "test", "valid"]:
        #print "SPLIT %s" % split
        split = x[split]
        #print len(split[0])
        wocount = 0
        for i in range(len(split[0])):
            #print ed[split[1][i][0]], ed[split[1][i][0]] in entidsnotinlabeldic
            #break
            if ed[split[1][i][0]] in entidsnotinlabeldic:
                #print " ".join(map(lambda x: wd[x] if x in wd else "", list(split[0][i]))), ed[split[1][i][0]]
                wocount += 1
            else:
                pass
                #print " ".join(map(lambda x: wd[x] if x in wd else "", list(split[0][i]))), labeldic[ed[split[1][i][0]]]

        #print wocount



if __name__ == "__main__":
    run()