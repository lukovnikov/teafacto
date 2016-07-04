import pickle, numpy as np
from IPython import embed

d = pickle.load(open("datamat.word.dmp.pkl"))
td = d["train"]
vd = d["valid"]
xd = d["test"]

wd = d["worddic"]
rwd = {v: k for k, v in wd.items()}
ld = d["labels"]
ed = d["entdic"]
red = {v: k for k, v in ed.items()}

# do the search results contain the true subject entity?
def searchcov(data):
    twong = []
    for i in range(len(data[0])):
        if data[1][i][0] not in data[2][i]:
            twong.append(i)
    return twong

def numcanstats(data):
    canc = {}
    for cans in data[2]:
        if len(cans) not in canc:
            canc[len(cans)] = 0
        canc[len(cans)] += 1
    return canc

def printidx(widxs):
    print " ".join(np.vectorize(lambda x: rwd[x] if x in rwd else "")(widxs))

print len(searchcov(td))
print len(searchcov(vd))
print len(searchcov(xd))

print numcanstats(td)
print numcanstats(vd)
print numcanstats(xd)