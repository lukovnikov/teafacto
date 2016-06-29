from teafacto.scripts.simplequestions.subjdet import loadlabels
import pickle


def run(labelp="labels.map", datap="datamat.word.pkl"):
    labeldic = loadlabels(labelp)
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
    for split in ["train", "test", "valid"]:
        print "SPLIT %s" % split
        split = x[split]
        #print len(split[0])
        wocount = 0
        for i in range(len(split[0])):
            #print ed[split[1][i][0]], ed[split[1][i][0]] in entidsnotinlabeldic
            #break
            if ed[split[1][i][0]] in entidsnotinlabeldic:
                print " ".join(map(lambda x: wd[x] if x in wd else "", list(split[0][i]))), ed[split[1][i][0]]
                wocount += 1
            else:
                print " ".join(map(lambda x: wd[x] if x in wd else "", list(split[0][i]))), labeldic[ed[split[1][i][0]]]

        print wocount



if __name__ == "__main__":
    run()