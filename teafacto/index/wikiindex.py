from whoosh.index import create_in, open_dir
from whoosh.fields import *
from whoosh.qparser import QueryParser, OrGroup
from whoosh.analysis import StemmingAnalyzer
import shelve, os, os.path
import nltk
#from teafacto.core.utils import FileHandler
from IPython import embed

import re


class WikipediaIndex(object):
    def __init__(self, dir="../../data/wikipedia/pagesidx/"):
        self.dir = dir
        #FileHandler.ensuredir(self.dir)
        self.ix = None
        self.searcher = None
        self.defaulturis = {"abstract": "http://downloads.dbpedia.org/2015-04/core-i18n/en/long-abstracts_en.nt.bz2"}

    def indexabstracts(self, source="../../data/wikipedia/long-abstracts_en.nt", atre=r"^<([^>]+)>\s<[^>]+>\s\"(.+)\"@en\s\.$"):
        atre = re.compile(atre)
        ana = StemmingAnalyzer()
        self.schema = Schema(title=ID(stored=True, unique=True), abstract=TEXT(analyzer=ana, stored=True))
        ix = create_in(self.dir, self.schema)
        #FileHandler.ensurefile(source, self.defaulturis["abstract"])
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

    def indexdump(self, source="../../data/wikipedia/ck12.txt", procs=3, limitmb=300, paragraphlevel=False):
        titlere = re.compile("^<\./(.+)\.txt>$")
        ana = StemmingAnalyzer()
        self.schema = Schema(title=ID(stored=True, unique=True), content=TEXT(analyzer=ana, stored=True))
        ix = create_in(self.dir, self.schema)
        try:
            with open(source) as sf:
                writer = ix.writer(procs=procs, limitmb=limitmb)
                c = 0
                currenttitle = None
                currentcontent = ""
                pc = 0
                for line in sf:
                    try:
                        m = titlere.match(line)
                        if m: # title
                            # flush previously accumulated title and content
                            if currenttitle is not None and len(currentcontent) > 0 and not paragraphlevel:
                                writer.add_document(title=currenttitle, content=currentcontent)
                            # set title, reset content
                            currenttitle = m.group(1).decode("unicode-escape")
                            currentcontent = ""
                            pc = 0
                        else: # content
                            if len(line) > 3:
                                currentcontent += line.decode("unicode-escape") + " "
                                if paragraphlevel:
                                    writer.add_document(title=currenttitle + "p%d" % pc, content=currentcontent)
                                    currentcontent = ""
                                pc += 1
                    except Exception as e:
                        print e, line
                # flush latest accumulated title and content
                if currenttitle is not None and len(currentcontent) > 0:
                    writer.add_document(title=currenttitle, content=currentcontent)
                writer.commit()
        except Exception as e:
            print e

    def _search(self, q="test", limit=20, field="content", orgroup=True):
        if self.ix is None:
            self.ix = open_dir(self.dir)
        if self.searcher is None:
            self.searcher = self.ix.searcher()
        ret = []
        rets = []
        try:
            query = QueryParser(field, self.ix.schema, group=OrGroup.factory(0.9)).parse(q)
            rets = self.searcher.search(query, limit=limit)
        except AttributeError as e:
            print e, q
        for r in rets:
            ret.append({"score": r.score, "content": r["content"], "title": r["title"]})
        return ret

    def search(self, q="test", limit=20):
        return self._search(q, limit, field="content")

    def searchtitle(self, t="test", limit=20):
        return self._search(t, limit, field="title")

    def getsentences(self, q="test", limit=20):
        sres = self.search(q, limit=limit)
        ret = []
        if sres is not None:
            for sre in sres:
                sresents = nltk.tokenize.sent_tokenize(sre["content"])
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

    wi = WikipediaIndex(dir="../../data/wikipedia/nparaidx/")
    #wi.indexdump(source="../../data/wikipedia/npages.txt", paragraphlevel=True)

    def srh(k):
        print "searching for: \"%s\"" % k
        sents = wi.search(k, limit=8)
        for sent in sents:
            print "%.3f - %s" % (sent["score"], sent["title"])# + sent["content"]
        print "\n"

    srh("color blindness")
    srh("mercury")
    srh("industry")
    srh("barack obama")
    srh("fumes")

    #ss = nltk.tokenize.sent_tokenize("hello, it's me. I was wondering.")

