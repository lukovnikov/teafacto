from whoosh.index import create_in, open_dir
from whoosh.fields import *
from whoosh.qparser import QueryParser
from whoosh.analysis import StemmingAnalyzer
import shelve, os, os.path
import nltk

import re
atre = re.compile(r"^<([^>]+)>\s<[^>]+>\s\"(.+)\"@en\s\.$")

class WikipediaIndex(object):
    def __init__(self, dir="../../data/wikipedia/idx/"):
        self.dir = dir

    def index(self, source="../../data/wikipedia/short-abstracts_en.nt"):
        ana = StemmingAnalyzer()
        self.schema = Schema(title=ID(stored=True, unique=True), abstract=TEXT(analyzer=ana, stored=True))
        ix = create_in(self.dir, self.schema)
        with open(source) as sf:
            writer = ix.writer(procs=3, limitmb=300)
            c = 0
            for line in sf:
                m = atre.match(line)
                if m:
                    title = m.group(1).decode("unicode-escape")
                    text = m.group(2).decode("unicode-escape")
                    writer.add_document(title=title, abstract=text)
                    c += 1
                    if c % 100000 == 0:
                        print c, text
                        print "committing"
                        writer.commit()
                        print "committed"
                        writer = ix.writer(procs=3, limitmb=300)
            writer.commit()

    def search(self, q="test", limit=20):
        ix = open_dir(self.dir)
        ret = []
        with ix.searcher() as searcher:
            query = QueryParser("abstract", ix.schema).parse(q)
            rets = searcher.search(query, limit=limit)
            for r in rets:
                ret.append({"title": r["title"], "abstract": r["abstract"]})
        return ret

    def getsentences(self, q="test", limit=20):
        sres = self.search(q, limit=limit)
        ret = []
        if sres is not None:
            for sre in sres:
                sresents = nltk.tokenize.sent_tokenize(sre["abstract"])
                ret.extend(sresents)
        return ret

    def populatedb(self, source="../../data/index/short-abstracts_en.nt", dbpath="../../data/index/abstracts.db"):
        db = shelve.open(dbpath, writeback=True)
        with open(source) as sf:
            c = 0
            for line in sf:
                m = atre.match(line)
                if m:
                    title = m.group(1).decode("unicode-escape")
                    text = m.group(2).decode("unicode-escape")
                    db[str(title)] = {"abstract": text}
                    c += 1
                    if c % 100000 == 0:
                        print c
                        db.sync()
        db.close()







if __name__ == "__main__":

    wi = WikipediaIndex()
    wi.index()

    sents = wi.getsentences("mercury", limit=20)
    for sent in sents:
        print sent

    #ss = nltk.tokenize.sent_tokenize("hello, it's me. I was wondering.")

