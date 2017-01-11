import re, random, numpy as np
from teafacto.query.lbd2tree import LambdaParser


# GENERATOR OF LAMBDA EXPRESSIONS GIVEN A NUMBER OF RELATIONS AND ENTITIES


class LambdaGrammar(object):
    def __init__(self, rules=None, symbols=None):
        self.symbols = symbols if symbols is not None else {}
        self.types = {}

    def __getitem__(self, item):    # item is string, symbol name
        if item not in self.symbols:
            if re.match("[A-Z]+", item):
                ret = NonTerminalSymbol(item)
            else:
                ret = TerminalSymbol(item)
            self.add_symbol(ret)
        return self.symbols[item]

    def add_symbol(self, sym):
        if sym.name not in self.symbols:
            self.symbols[sym.name] = sym
        else:
            raise KeyError("symbol already in grammar")

    def __add__(self, other):
        if isinstance(other, LambdaSymbol):
            self.add_symbol(other)
            return self
        else:
            raise NotImplementedError("not supported: {}".format(type(other)))

    def __repr__(self):
        return "lambda grammar"

    def __str__(self):
        ret = "SYMBOLS================================\n"
        ret += "\n".join([str(sym) for k, sym in self.symbols.items()])
        return ret

    def parse_rules(self, s):
        # rules must be of form HEAD -> BODY | BODY | ... ;
        g = self
        def f(accu):
            m = re.match("^([A-Z]+)\s?->(\s?[^;]+\s?)\s?;$", accu)
            head = m.group(1)
            body = m.group(2)
            headsym = g[head]
            bodysplits = body.split("|")
            for bodysplit in bodysplits:
                bodysplit = bodysplit.lstrip(" ").rstrip(" ")
                seq = bodysplit.split()
                acc = []
                for seqelem in seq:
                    sym = self[seqelem]
                    acc.append(sym)
                self.add_rule(LambdaRule(headsym, acc))
        self._parse_loop(s, f)

    def add_rule(self, rule):
        headsym = self[rule.src.name]
        headsym.add_rules(rule)

    def add_rule_str(self, fro, to):
        r = LambdaRule(self[fro], [self[to]])
        self.add_rule(r)

    def _parse_loop(self, d, f):
        # rules must be of form HEAD -> BODY | BODY | ... ;
        ss = d.split("\n")
        accu = ""
        for split in ss:
            m = re.match("^([^#]*)(#.+)?$", split)
            t = m.group(1)
            t = t.rstrip(" \t").lstrip(" \t")
            accu += t
            if len(t) > 0 and t[-1] == ";":
                f(accu)
                accu = ""

    def parse_info(self, s):
        def f(accu):
            accu = accu.rstrip(" ;\t").lstrip(" \t")
            splits = accu.split("<")
            head = splits[0].rstrip(" \t").lstrip(" \t")
            body = splits[1]
            if body[0] == "=":  # type expression
                body = body[1:]
                headsym = self[head]
                t = self.get_type(body)
                assert(isinstance(headsym, TerminalSymbol))
                headsym.typ = t
            else:               # subtype expression
                t = self.get_type(body)
                h = self.get_type(head)
                h.supertype = t

        self._parse_loop(s, f)

    def get_type(self, s):
        s = s.lstrip(" \t").rstrip(" \t")
        ss = s.split("->")
        if len(ss) == 1:    # unary type
            if s not in self.types:
                self.types[s] = UnaryType(s)
            return self.types[s]
        elif len(ss) == 2:  # binary type
            fro = map(lambda x: self.get_type(x.rstrip(" ").lstrip(" ")), ss[0].lstrip(" ").rstrip(" ").split("|"))
            to = map(lambda x: self.get_type(x.rstrip(" ").lstrip(" ")), ss[1].lstrip(" ").rstrip(" ").split("|"))
            if s not in self.types:
                self.types[s] = BinaryType(fro, to)
            return self.types[s]
        else:
            raise NotImplementedError("higher-order types not supported")


class LambdaSymbol(object):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return "Lambda Symbol"

    def __str__(self):
        return self.name + "---"


class NonTerminalSymbol(LambdaSymbol):
    def __init__(self, name):
        super(NonTerminalSymbol, self).__init__(name)
        self.rules = set()

    def add_rules(self, *rules):
        for rule in rules:
            if rule not in self.rules:
                self.rules.add(rule)

    def __repr__(self):
        return "nonterm"

    def __str__(self):
        return self.name + ":\n" + "\n".join(["\t{}".format(x) for x in self.rules]) if len(self.rules) > 0 else self.name


class TerminalSymbol(LambdaSymbol):
    def __init__(self, name, typ=None):
        super(TerminalSymbol, self).__init__(name)
        self.typ = typ

    def __repr__(self):
        return self.name

    def __str__(self):
        return "{}[{}]".format(self.name, self.typ) if self.typ is not None else self.name


class VarSymbol(TerminalSymbol):
    def __init__(self, name, typ=None):
        super(VarSymbol, self).__init__(name, typ=typ)


class SymbolType(object):
    pass


class UnaryType(SymbolType):
    def __init__(self, name):
        self.name = name
        self._supertype = None

    @property
    def supertype(self):
        return self._supertype

    @supertype.setter
    def supertype(self, st):
        self._supertype = st

    def __repr__(self):
        return self.name


class BinaryType(SymbolType):
    def __init__(self, fro, to):    # both args are Unary Types or collections
        self.fro = fro
        self.to = to

    def __repr__(self):
        return "|".join(map(str, self.fro)) + " -> " + "|".join(map(str, self.to))


class LambdaRule(object):
    def __init__(self, src, dst, p=1.0, **kw):
        super(LambdaRule, self).__init__(**kw)
        self.src = src
        self.dst = dst
        self.p = p

    def __str__(self):
        return "{} -> {}".format(self.src.name, " ".join(map(lambda x: x.name, self.dst)))

    def __repr__(self):
        return str(self)


class Generator(object):
    def __init__(self, grammar, start="S", limit=12):
        self.start = start
        self.grammar = grammar
        self.limit = limit

    def generate(self):
        cursym = self.grammar[self.start]
        ret = None
        while ret is None or len(ret) < 3:
            ret = self.generate_rec(cursym, set(), depth=self.limit)
        return ret

    def generate_rec(self, cursym, vars, depth=1):
        if depth == 0:
            return None
        if isinstance(cursym, TerminalSymbol):  # terminal, do nothing
            return [cursym]
        elif isinstance(cursym, NonTerminalSymbol):     # expand
            if cursym.name == "CVAR":       # create var
                for i in xrange(0, 99):
                    ret = VarSymbol("${}".format(i))
                    if ret.name not in set(map(lambda x: x.name, list(vars))):
                        break
                return [ret]
            elif cursym.name == "VAR":
                ret = random.choice(list(vars))
                return [ret]
            if cursym.name == "NUM" and False:
                return [TerminalSymbol(str(random.randint(0, 100)) + ":i")]
            rules = cursym.rules
            if len(rules) == 0:
                print "no rules for non terminal: " + str(cursym)
            rules = list(rules)
            probs = np.asarray([rule.p for rule in rules])
            probs = probs / np.sum(probs)
            rule = np.random.choice(list(rules), p=probs)
            dst = rule.dst
            ret = []
            for dste in dst:
                r = self.generate_rec(dste, set().union(vars), depth=depth-1)
                if r is None:
                    return r
                if len(r) == 1 and isinstance(r[0], VarSymbol):
                    vars = {r[0],}.union(vars)
                ret.extend(r)
            return ret
        else:
            raise Exception("???")


def estimate_weights(grammar, parser, path):
    curline = ""
    for line in open(path):
        if len(curline) == 0:
            curline = line[:-1]
        else:
            if line == "\n":
                #print curline
                #d.append(""+curline)
                curline = ""
            elif line[:2] == "//":
                pass
            else:
                line = line[:-1]
                curline = "{}\t{}".format(curline, line)
                lbd = curline.split("\t")[1]
                parsed = parser.parse(lbd)


# deterministic CFG parser for unambiguous lambda language
class DCFGParser(object):
    def __init__(self, grammar):
        self.grammar = grammar

    def parse(self, s):
        stack = []
        for rule in self.grammar.rules:
            pass


class ParseNode(object):
    def __init__(self):
        pass


def get_terminals_from_data(path, parser):
    curline = ""
    ents = set()
    rels = set()
    rellits = set()
    types = set()
    for line in open(path):
        if len(curline) == 0:
            curline = line[:-1]
        else:
            if line == "\n":
                #print curline
                #d.append(""+curline)
                curline = ""
            elif line[:2] == "//":
                pass
            else:
                line = line[:-1]
                curline = "{}\t{}".format(curline, line)
                lbd = curline.split("\t")[1]
                parsed = parser.parse(lbd)
                q = [parsed]
                while len(q) > 0:
                    qt = q.pop()
                    m = re.match("(\w+):(\w+)", qt.name)
                    if m:
                        if len(qt.children) == 1:       # type or relsel
                            done = False
                            if qt.parent is not None and \
                                    qt.parent.name in ("argmax", "argmin", "sum"):
                                pos = 0
                                while pos < 3:
                                    if qt.parent.children[pos] == qt:
                                        break
                                    pos += 1
                                if pos == 2:
                                    rellits.add(qt.name)
                                    done = True
                            if not done and \
                                    re.match("\$\d", qt.children[0].name) and \
                                    (qt.parent is not None
                                     and not re.match("\w+:\w+", qt.parent.name) and not qt.parent.name in ("=", ">", "<")):
                                if qt.name[-1] == "i":
                                    pass
                                types.add(qt.name)
                            elif not done:
                                if m.group(2) == "i":
                                    rellits.add(qt.name)
                                else:
                                    rels.add(qt.name)
                        elif len(qt.children) == 2:     # relation
                            rels.add(qt.name)
                        else:
                            ents.add(qt.name)
                    q.extend(qt.children)
                #print parsed
    return ents, rels, rellits, types


if __name__ == "__main__":
    # non-terminals:  ENT/NUM
    # terminals: specific entities/values, language elements (lambda, ...)
    # types and relation domain/range necessary
    # example generations:
    #   EXPR -> CITY -> (REL[STATE-CITY] STATE) -> (capital:c Ohio:s)
    #   S -> INT -> (REL[STATE-NUM] STATE) -> (population:i Ohio:s)
    #   S -> STATE -> (lambda $0 e EXPR)
    #             -> (lambda $0 e (and EXPR EXPR))
    #             -> (lambda $0 e (and (
    # (lambda $0 e (and (state:t $0) (loc:t dallas_tx:c $0)))
    lbdrules = """
        S -> ENT | SEL ;
        COND -> ( TYPE VAR )
                | ( REL VAR ENT ) | ( REL ENT VAR ) | ( REL VAR VAR )
                | ( and COND COND )
                | ( not COND )
                | ( exists CVAR COND )
                | ( COMP SELLITORNUM SELLITORNUM );
        ENT ->  SENT
                | ( the CVAR COND )
                | ( argmax CVAR COND SELLITVAR )
                | ( argmin CVAR COND SELLITVAR ) ;
        SEL ->  ( REL ENT );
        SELLIT -> SELLITVAR | ( count CVAR COND ) | ( sum CVAR COND SEL ) ;
        SELLITVAR -> ( RELLIT VAR ) ;
        SELLITORNUM -> SELLIT | NUM ;
        VAR -> $ ;
        COMP -> = | > | < ;
        CVAR -> $ ;
        NUM -> 0:i ;
   """
    terminallbdrules = """
        SENT -> ohio:s | washington:s | mississippi_river:r ;
        REL -> capital:c | capital:t | next_to:t | equals:t ;
        RELLIT -> population:i | size:i ;
        TYPE -> state:t | city:t ;
    """
    typeinfo = """
        ohio:s <= state:t ;
        washington:s <= state:t ;
        seattle:c <= city:t ;
        capital:c <= state:t | country:t -> city:t ;
        population:i <= country:t | state:t | city:t -> int ;
        city:t < place:t ;
    """
    g = LambdaGrammar()
    g.parse_rules(lbdrules)
    p = "../../data/semparse/geoquery.lbd.dev"
    parser = LambdaParser()
    ents, rels, rellits, types = get_terminals_from_data(p, parser)
    tp = "../../data/semparse/geoquery.lbd.test"
    tents, trels, trellits, ttypes = get_terminals_from_data(tp, parser)
    print map(len, [tents.difference(ents), trels.difference(rels), trellits.difference(rellits), ttypes.difference(types)])
    ents = ents.union(tents)
    rels = rels.union(trels)
    rellits = rellits.union(trellits)
    types = types.union(ttypes)
    for ent in ents:
        g.add_rule_str("SENT", ent)
    for rel in rels:
        g.add_rule_str("REL", rel)
    for rellit in rellits:
        g.add_rule_str("RELLIT", rellit)
    for type in types:
        g.add_rule_str("TYPE", type)
    g.parse_info(typeinfo)
    #print g
    gen = Generator(g)
    #print "\n"
    from teafacto.util import ticktock as TT
    tt = TT()
    tt.tick()
    k = 50000
    outp = "../../data/semparse/geoquery.lbd.autogen"
    with open(outp, "w") as f:
        parser = LambdaParser()
        for i in range(k):
            x = " ".join(map(lambda x: x.name, gen.generate()))
            y = parser.parse(x)
            if y.name == "the":
                y.name = "lambda"
                y.children = (y.children[0], "e", y.children[1])
            print y.greedy_linearize()
            print y
            f.write("{}\n{}\n\n".format(y.greedy_linearize(), str(y)))
    tt.tock("generated {} samples".format(k))

    #p = "../../data/semparse/geoquery.lbd.dev"
    #parser = LambdaParser()
    #estimate_weights(g, parser, p)
