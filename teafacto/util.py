import collections, inspect, argparse, dill as pkl, os, numpy as np, pandas as pd, sys
from datetime import datetime as dt
import re, unidecode, nltk
from nltk.corpus import stopwords


def loadlexidtsv(path, numwords=10, numchars=30):
    with open(path) as f:
        allgloveids = []    # 2D
        allcharmats = []    # 3D
        allfbids = []       # 1D

        c = 0
        for line in f:
            try:
                ns = line[:-1].split("\t")
                gloveids = map(int, ns[0].split(" "))
                charmat = []
                charsplits = ns[1].split(", ")
                charmat.extend([[int(y) if len(y) > 0 else 0 for y in x.split(" ")] for x in charsplits])
                fbid = int(ns[2])
                allgloveids.append(gloveids)
                allcharmats.append(charmat)
                allfbids.append(fbid)
                if c % 1e6 == 0:
                    print "%.0fM" % (c/1e6)
                c += 1
            except Exception, e:
                print line
                raise e
        allgloveids, allcharmats, allfbids = makenpfrom(allgloveids, allcharmats, allfbids, dtype="int32", numwords=numwords, numchars=numchars)
        return allgloveids, allcharmats, allfbids


def makenpfrom(tomat, toten, tovec, dtype="int32", numwords=15, numchars=30):
    i = 0
    delc = 0
    truncwc = 0
    assert(len(tomat) == len(toten) and len(toten) == len(tovec))
    while i < len(tomat):
        assert(len(tomat[i]) == len(toten[i]))
        if len(tomat[i]) > numwords:   # drop phrases longer than <numwords> words
            #print tovec[i]
            tomat[i] = tomat[i][:numwords]
            toten[i] = toten[i][:numwords]
            delc += 1
        else:
            tomat[i].extend([0]*(numwords - len(tomat[i])))
        j = 0
        while j < len(toten[i]):
            if len(toten[i][j]) > numchars:     # if word is too long, truncate
                toten[i][j] = toten[i][j][:numchars]
                truncwc += 1
            else:
                toten[i][j].extend([0]*(numchars-len(toten[i][j])))
            j += 1
        torep = [[0]*numchars]
        toten[i].extend(torep*(numwords-len(toten[i])))
        i += 1
        if i % 1e6 == 0:
            print "%.0fM, %d, %d" % ((i/1e6), delc, truncwc)
    print i, delc, truncwc
    return np.asarray(tomat, dtype=dtype), np.asarray(toten, dtype=dtype), np.asarray(tovec, dtype=dtype)


def unstructurize(x, i=None):
    if i is None:
        i = []
    if isinstance(x, dict):
        out = {}
        for key in x.keys():
            ret, i = unstructurize(x[key], i)
            out[key] = ret
    elif isinstance(x, (list, tuple)):
        out = []
        for elem in x:
            ret, i = unstructurize(elem, i)
            out.append(ret)
        if isinstance(x, tuple):
            out = tuple(out)
    elif isinstance(x, set):
        raise Exception("sets not supported")
    else:
        out = len(i)
        i.append(x)
    return out, i


def restructurize(n, f):
    if isinstance(n, dict):
        out = {}
        for key in n.keys():
            out[key] = restructurize(n[key], f)
    elif isinstance(n, (list, tuple)):
        out = []
        for elem in n:
            out.append(restructurize(elem, f))
        if isinstance(n, tuple):
            out = tuple(out)
    elif isinstance(n, set):
        raise Exception("sets not supported")
    else:
        out = f[n]
    return out





class ticktock(object):
    def __init__(self, prefix="-", verbose=True):
        self.prefix = prefix
        self.verbose = verbose
        self.state = None
        self.perc = None
        self.prevperc = None
        self._tick()

    def tick(self, state=None):
        if self.verbose and state is not None:
            print "%s: %s" % (self.prefix, state)
        self._tick()

    def _tick(self):
        self.ticktime = dt.now()

    def _tock(self):
        return (dt.now() - self.ticktime).total_seconds()

    def progress(self, x, of, action="", live=False):
        if self.verbose:
            self.perc = int(round(100. * x / of))
            if self.perc != self.prevperc:
                if action != "":
                    action = " " + action + " -"
                topr = "%s:%s %d" % (self.prefix, action, self.perc) + "%"
                if live:
                    self._live(topr)
                else:
                    print(topr)
                self.prevperc = self.perc

    def tock(self, action=None, prefix=None):
        duration = self._tock()
        if self.verbose:
            prefix = prefix if prefix is not None else self.prefix
            action = action if action is not None else self.state
            print "%s: %s in %s" % (prefix, action, self._getdurationstr(duration))
        return self

    def msg(self, action=None, prefix=None):
        if self.verbose:
            prefix = prefix if prefix is not None else self.prefix
            action = action if action is not None else self.state
            print "%s: %s" % (prefix, action)
        return self

    def _getdurationstr(self, duration):
        if duration >= 60:
            duration = int(round(duration))
            seconds = duration % 60
            minutes = (duration // 60) % 60
            hours = (duration // 3600) % 24
            days = duration // (3600*24)
            acc = ""
            if seconds > 0:
                acc = ("%d second" % seconds) + ("s" if seconds > 1 else "")
            if minutes > 0:
                acc = ("%d minute" % minutes) + ("s" if minutes > 1 else "") + (", " + acc if len(acc) > 0 else "")
            if hours > 0:
                acc = ("%d hour" % hours) + ("s" if hours > 1 else "") + (", " + acc if len(acc) > 0 else "")
            if days > 0:
                acc = ("%d day" % days) + ("s" if days > 1 else "") + (", " + acc if len(acc) > 0 else "")
            return acc
        else:
            return ("%.3f second" % duration) + ("s" if duration > 1 else "")

    def _live(self, x, right=None):
        if right:
            try:
                #ttyw = int(os.popen("stty size", "r").read().split()[1])
                raise Exception("qsdf")
            except Exception:
                ttyw = None
            if ttyw is not None:
                sys.stdout.write(x)
                sys.stdout.write(right.rjust(ttyw - len(x) - 2) + "\r")
            else:
                sys.stdout.write(x + "\t" + right + "\r")
        else:
            sys.stdout.write(x + "\r")
        sys.stdout.flush()

    def live(self, x):
        if self.verbose:
            self._live(self.prefix + ": " + x, "T: %s" % self._getdurationstr(self._tock()))

    def stoplive(self):
        if self.verbose:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()


def argparsify(f, test=None):
    args, _, _, defaults = inspect.getargspec(f)
    assert(len(args) == len(defaults))
    parser = argparse.ArgumentParser()
    i = 0
    for arg in args:
        argtype = type(defaults[i])
        if argtype == bool:     # convert to action
            if defaults[i] == False:
                action="store_true"
            else:
                action="store_false"
            parser.add_argument("-%s" % arg, "--%s" % arg, action=action, default=defaults[i])
        else:
            parser.add_argument("-%s"%arg, "--%s"%arg, type=type(defaults[i]))
        i += 1
    if test is not None:
        par = parser.parse_args([test])
    else:
        par = parser.parse_args()
    kwargs = {}
    for arg in args:
        if getattr(par, arg) is not None:
            kwargs[arg] = getattr(par, arg)
    return kwargs


def argprun(f, sigint_shell=True, **kwargs):   # command line overrides kwargs
    def handler(sig, frame):
        # find the frame right under the argprun
        print "custom handler called"
        original_frame = frame
        current_frame = original_frame
        previous_frame = None
        stop = False
        while not stop and current_frame.f_back is not None:
            previous_frame = current_frame
            current_frame = current_frame.f_back
            if "_FRAME_LEVEL" in current_frame.f_locals \
                and current_frame.f_locals["_FRAME_LEVEL"] == "ARGPRUN":
                stop = True
        if stop:    # argprun frame found
            l = previous_frame.f_locals     # f-level frame locals
            stopprompt = False
            while not stopprompt:
                whattodo = raw_input("(s)hell, (k)ill\n>>")
                if whattodo == "s":
                    embed()
                elif whattodo == "k":
                    "Killing"
                    sys.exit()
                else:
                    stopprompt = True

    if sigint_shell:
        _FRAME_LEVEL="ARGPRUN"
        prevhandler = signal.signal(signal.SIGINT, handler)
    try:
        f_args = argparsify(f)
        for k, v in kwargs.items():
            if k not in f_args:
                f_args[k] = v
        f(**f_args)

    except KeyboardInterrupt:
        print("Interrupted by Keyboard")


def inp():
    return raw_input("Press ENTER to continue:\n>>> ")


def issequence(x):
    return isinstance(x, collections.Sequence) and not isinstance(x, basestring)


def isnumber(x):
    return isinstance(x, float) or isinstance(x, int)


def isstring(x):
    return isinstance(x, basestring)


def iscallable(x):
    return hasattr(x, "__call__")


def isfunction(x):
    return iscallable(x)


def getnumargs(f):
    return len(inspect.getargspec(f).args)


class Saveable(object):
    def __init__(self, autosave=False, **kw):
        super(Saveable, self).__init__(**kw)
        self._autosave = autosave

    ############# Saving and Loading #################"
    def getdefaultsavepath(self):
        dir = "../../saves/"
        if not os.path.exists(os.path.join(os.path.dirname(__file__), dir)):
            os.makedirs(os.path.join(os.path.dirname(__file__), dir))
        dfile = os.path.join(os.path.dirname(__file__), dir+"%s.%s" %
                             (self.printname, dt.now().strftime("%Y-%m-%d=%H:%M")))
        return dfile

    @property
    def printname(self):
        return self.__class__.__name__

    def save(self, filepath=None):
        if filepath is None:
            filepath = self.getdefaultsavepath() + ".auto"
        with open(filepath, "w") as f:
            pkl.dump(self, f)
        return filepath

    # TODO: WILL FREEZE/UNFREEZE WORK WITH GPU?
    def freeze(self):
        return pkl.dumps(self)

    @staticmethod
    def unfreeze(dumps):
        return pkl.loads(dumps)

    @staticmethod
    def load(filepath):
        with open(filepath) as f:
            ret = pkl.load(f)
        return ret

    @property
    def autosave(self): # for saving after each iter
        self._autosave = True
        return self


class DataCollection():
    def __init__(self, **datas):
        self.datas = datas

    def update(self, **datas):
        self.datas.update(datas)

    def __getattr__(self, item):
        if item in self.datas:
            return self.datas[item]
        else:
            raise AttributeError


class DataSet():
    def __init__(self, data, gold):
        self.data = data
        self.gold = gold

    def __repr__(self):
        return str(self.data) + "\n" + str(self.gold)


class StringMatrix():
    protectedwords = ["<MASK>", "<RARE>", "<START>", "<END>"]
    def __init__(self, maxlen=None, freqcutoff=0, topnwords=None, indicate_start_end=False):
        self._strings = []
        self._wordcounts = dict(zip(self.protectedwords, [0]*len(self.protectedwords)))
        self._dictionary = dict(zip(self.protectedwords, range(len(self.protectedwords))))
        self._rd = None
        self._next_available_id = len(self._dictionary)
        self._maxlen = 0
        self._matrix = None
        self._max_allowable_length = maxlen
        self._rarefreq = freqcutoff
        self._topnwords = topnwords
        self._indic_s_e = indicate_start_end
        self._rarewords = set()

    @property
    def numwords(self):
        return len(self._dictionary)

    @property
    def numrare(self):
        return len(self._rarewords)

    @property
    def matrix(self):
        if self._matrix is None:
            raise Exception("finalize first")
        return self._matrix

    def d(self, x):
        return self._dictionary[x]

    def rd(self, x):
        return self._rd[x]

    def pp(self, matorvec):
        def pp_vec(vec):
            return " ".join([self.rd(x) if x in self._rd else "<UNK>" for x in vec if x != self.d("<MASK>")])
        ret = []
        if matorvec.ndim == 2:
            for vec in matorvec:
                ret.append(pp_vec(vec))
        else:
            return pp_vec(matorvec)
        return ret

    def add(self, x):
        tokens = tokenize(x)
        tokens = tokens[:self._max_allowable_length]
        if self._indic_s_e:
            tokens = ["<START>"] + tokens + ["<END>"]
        self._maxlen = max(self._maxlen, len(tokens))
        tokenidxs = []
        for token in tokens:
            if token not in self._dictionary:
                self._dictionary[token] = self._next_available_id
                self._next_available_id += 1
                self._wordcounts[token] = 0
            self._wordcounts[token] += 1
            tokenidxs.append(self._dictionary[token])
        self._strings.append(tokenidxs)

    def finalize(self):
        ret = np.zeros((len(self._strings), self._maxlen), dtype="int32")
        for i, string in enumerate(self._strings):
            ret[i, :len(string)] = string
        self._matrix = ret
        self._do_rare()
        self._rd = {v: k for k, v in self._dictionary.items()}

    def _do_rare(self):
        sortedwordidxs = [self.d(x) for x in self.protectedwords] + \
                         ([self.d(x) for x, y
                          in sorted(self._wordcounts.items(), key=lambda (x,y): y, reverse=True)
                          if y >= self._rarefreq and x not in self.protectedwords][:self._topnwords])
        transdic = zip(sortedwordidxs, range(len(sortedwordidxs)))
        transdic = dict(transdic)
        self._rarewords = {x for x in self._dictionary.keys() if self.d(x) not in transdic}
        rarewords = {self.d(x) for x in self._rarewords}
        self._numrare = len(rarewords)
        transdic.update(dict(zip(rarewords, [self.d("<RARE>")]*len(rarewords))))
        # translate matrix
        self._matrix = np.vectorize(lambda x: transdic[x])(self._matrix)
        # change dictionary
        self._dictionary = {k: transdic[v] for k, v in self._dictionary.items() if self.d(k) in sortedwordidxs}


def tokenize(s):
    s = s.decode("utf-8")
    s = unidecode.unidecode(s)
    s = re.sub("[-_\{\}/]", " ", s)
    s = s.lower()
    tokens = nltk.word_tokenize(s)
    s = re.sub("`", "'", s)
    return tokens


from IPython import embed
import signal


def handleSIGINT():
    def handler(signal, frame):
        stop = False
        while not stop:
            choice = raw_input("Execution interrupted: (c)ontinue, (i)python shell, (k)ill")
            if choice == "i":
                embed()
            elif choice == "k":
                raise KeyboardInterrupt()
            elif choice == "c":
                pass
                stop = True
            else:
                print "option not supported"
    print "handling KI"
    signal.signal(signal.SIGINT, handler)



