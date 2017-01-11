import re, numpy as np

class CFG(object):
    def __init__(self):
        self.rules = []

    def __repr__(self):
        return "CFG"

    def __str__(self):
        ret = "RULES=========\n"
        ret += "\n".join(map(str, self.rules))
        return ret

    def parse_rules(self, s):
        # rules must be of form "HEAD -> BODY | BODY | ... ;"
        g = self
        def f(accu):
            m = re.match("^([A-Z]+)\s?->(\s?[^;]+\s?)\s?;$", accu)
            head = m.group(1)
            body = m.group(2)
            headsym = head
            bodysplits = body.split("|")
            for bodysplit in bodysplits:
                bodysplit = bodysplit.lstrip(" ").rstrip(" ")
                seq = bodysplit.split()
                acc = []
                for seqelem in seq:
                    sym = seqelem
                    acc.append(sym)
                g.rules.append(CFGRule(headsym, acc))
        self._parse_loop(s, f)

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


class CFGRule(object):
    def __init__(self, src, dst, w=0.0):
        self.src = src
        self.dst = dst
        self.w = w

    def __str__(self):
        return "{}: {} -> {}".format(self.w, self.src, " ".join(map(str, self.dst)))

    def __repr__(self):
        return str(self)


class DCFGParser(object):       # eager leftmost derivation
    def __init__(self, cfg):
        self.cfg = cfg

    def _prep_input(self, s):
        return [ParseNode(x) for x in s.split()]

    def _rule_body_match(self, rule, stack):
        if len(stack) < len(rule.dst):
            return False
        s = stack[-len(rule.dst):]
        return reduce(lambda a, b: a and b, map(lambda (x, y): x == y.symbol, zip(rule.dst, s)))

    def _rule_body_execute(self, rule, stack):
        if not self._rule_body_match(rule, stack):
            raise Exception("cannot reduce stack that doesn't match")
        s = stack[-len(rule.dst):]
        pn = ParseNode(rule.src, s, rule=rule)
        del stack[-len(rule.dst):]
        stack.append(pn)

    def parse(self, s):     # finds leftmost derivation (should be unique for DCFG)
        ss = self._prep_input(s)        # parsenodes
        stack = []      # start with empty stack
        while len(ss) > 0:
            tryreduce = True
            while tryreduce:    # try reduce
                tryreduce = False
                for rule in self.cfg.rules:     # check all rules
                    if self._rule_body_match(rule, stack):     # if rule matches stack
                        self._rule_body_execute(rule, stack)     # apply rule on stack (reduce)
                        tryreduce = True        # don't shift yet
                        break
            stack.append(ss.pop())          # shift
        if len(stack) == 1:
            print "ACCEPT"
        else:
            raise Exception("grammar does not accept given string")


class ParseNode(object):
    def __init__(self, symbol, children=[], rule=None):
        self.symbol = symbol
        self.children = children
        self.rule = rule



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
                | ( exists CVAR COND )
                | ( = SEL SEL ) | ( = SEL NUM ) | ( = NUM SEL )
                | ( > SEL SEL ) | ( > SEL NUM ) | ( > NUM SEL );
        ENT ->  SENT
                | ( the CVAR COND )
                | ( argmax CVAR COND SEL )
                | ( argmin CVAR COND SEL ) ;
        SEL ->  ( REL ENT ) | ( count CVAR COND );
        VAR -> $ ;
        CVAR -> $ ;
        NUM -> 0:i ;
    """
    terminallbdrules = """
        SENT -> ohio:s | washington:s | mississippi_river:r ;
        REL -> capital:c | population:i ;
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
    g = CFG()
    g.parse_rules(lbdrules)
    g.parse_rules(terminallbdrules)
    print g

    s = "(lambda $0 (capital:t hawaii:s $0))"
    p = DCFGParser(g)
    p.parse(s)