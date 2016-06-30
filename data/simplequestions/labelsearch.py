import elasticsearch
from teafacto.util import tokenize

def index(labelp="labels.map", host="drogon", index="simplequestions_labels"):
    es = elasticsearch.Elasticsearch(hosts=[host])
    i = 1
    for line in open(labelp):
        k, v = line[:-1].split("\t")
        es.index(index=index, doc_type="labelmap", id=i,
                 body={"label": v, "fbid": k})
        if i % 1000 == 0:
            print i
        i += 1
    print "indexed labels"

def search(query, top=10, index="simplequestions_labels", host="drogon"):
    es = elasticsearch.Elasticsearch(hosts=[host])
    res = es.search(index=index, q="label:%s" % query, size=top)
    acc = {}
    for r in res["hits"]["hits"]:
        _merge(acc, {r["_source"]["fbid"]: (r["_score"], r["_source"]["label"])})
    return acc


def _merge(acc, d):
    for k, v in d.items():
        if k in acc:
            if acc[k][0] < v[0]:
                acc[k] = v
        else:
            acc[k] = v


def searchsentence(s, top=10, index="simplequestions_labels", host="drogon", topsize=5):
    es = elasticsearch.Elasticsearch(hosts=[host])
    s = tokenize(s)
    cans = {}
    i = 0
    ngrams = set()
    while i < len(s) - 1:
        j = i + 1
        while j < min(len(s), i + topsize + 1):
            ngram = " ".join(s[i:j])
            j += 1
            ngrams.add(ngram)
        i += 1
    print ngrams
    for ngram in ngrams:
        ngramres = search(ngram, top=top)
        _merge(cans, ngramres)
    return cans


if __name__ == "__main__":
    #print search("e", top=30)
    res = searchsentence("to what release does the release track cardiac arrest come from", top=50)
    sres = sorted(res.items(), key=lambda (x, y): y[0], reverse=True)
    for x in sres:
        print x
    print len(sres)
