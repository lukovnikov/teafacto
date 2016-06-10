#!/usr/bin/env python
# -*- coding: utf-8 -*-

from unittest import TestCase
from teafacto.util import tokenize


class TestGetwords(TestCase):
    def test_split(self):
        s = "What's plàza-midwood  (wéèр) a type of rock-'n-rolla... at didn't o'clock?"
        ret = tokenize(s)
        print ret

