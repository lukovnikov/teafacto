import elasticsearch, re, sys
from teafacto.util import tokenize, argprun


class SimpleQuestionsLabelIndex(object):
    def __init__(self, host="drogon", index="simplequestions_labels"):
        self.host = host
        self.indexp = index

    def index(self, labelp="labels.map"):
        es = elasticsearch.Elasticsearch(hosts=[self.host])
        i = 1
        for line in open(labelp):
            k, v = line[:-1].split("\t")
            es.index(index=self.indexp, doc_type="labelmap", id=i,
                     body={"label": " ".join(tokenize(v)), "fbid": k})
            if i % 1000 == 0:
                print i
            i += 1
        print "indexed labels"

    def search(self, query, top=10):
        es = elasticsearch.Elasticsearch(hosts=[self.host])
        res = es.search(index=self.indexp, q="label:%s" % query, size=top)
        acc = {}
        for r in res["hits"]["hits"]:
            self._merge(acc, {r["_source"]["fbid"]: (r["_score"], r["_source"]["label"])})
        return acc

    @staticmethod
    def _merge(acc, d):
        for k, v in d.items():
            if k in acc:
                if acc[k][0] < v[0]:
                    acc[k] = v
            else:
                acc[k] = v

    def searchsentence(self, s, top=10, topsize=None):
        s = tokenize(s)
        ngrams = self.getallngrams(s, topsize)
        return self.searchallngrams(ngrams, top)

    def getallngrams(self, s, topsize=None):
        topsize = len(s) if topsize is None else topsize
        ngrams = set()
        i = 0
        while i < len(s):
            j = i + 1
            while j <= min(len(s), i + topsize):
                ngram = " ".join(s[i:j])
                j += 1
                ngrams.add(ngram)
            i += 1
        return ngrams

    def searchallngramso(self, ngrams, top=10):
        cans = {}
        for ngram in ngrams:
            ngram = '"%s"' % ngram
            ngramres = self.search(ngram, top=top)
            self._merge(cans, ngramres)
        return cans

    def searchallngrams(self, ngrams, top=10):
        #print ngrams
        es = elasticsearch.Elasticsearch(hosts=[self.host])
        searchbody = []
        header = {"index": self.indexp, "type": "labelmap"}
        for ngram in ngrams:
            ngram = '"%s"' % ngram
            body = {"from": 0, "size": top,
                    "query": {"match_phrase": {"label": ngram}}}
            searchbody.append(header)
            searchbody.append(body)
        ngramres = es.msearch(body=searchbody)
        cans = {}
        for response in ngramres["responses"]:
            try:
                for r in response["hits"]["hits"]:
                    self._merge(cans, {r["_source"]["fbid"]: (r["_score"], r["_source"]["label"])})
            except KeyError, e:
                print response
        return cans


def run(index=False, indexp="labels.map"):
    idx = SimpleQuestionsLabelIndex(host="localhost", index="simplequestions_labels")
    if index is True and indexp is not None:
        idx.index(labelp=indexp)
        sys.exit()
    #res = idx.search("e", top=10)
    res = idx.searchsentence("dutton adult", top=10)
    sres = sorted(res.items(), key=lambda (x, y): y[0], reverse=True)
    for x in sres:
        print x
    print len(sres)



if __name__ == "__main__":
    argprun(run)