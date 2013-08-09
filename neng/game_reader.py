#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

# Copyright (C) 2013 Petr Šebek

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHERWISE
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import shlex
from operator import mul
import os.path

import numpy as np
from functools import reduce


class GameReader(object):

    '''
    Read games from different file formats (.nfg payoff, .nfg outcome), see
    http://www.gambit-project.org/doc/formats.html for more information.
    '''
    def __init__(self):
        self.game = {}

    def readStr(self, string):
        """
        Base function that convert text to tokens a determine which

        :param string: string with nfg formated text
        :type string: str
        :return: dictionary with game informations
        :rtype: dict
        :raise: Exception, if the string is not in specified format
        """
        self.game.clear()
        self.tokens = shlex.split(string)
        preface = ["NFG", "1", "R"]
        if self.tokens[:3] != preface:
            raise Exception("Input string is not valid nfg format")
        self.game['name'] = self.tokens[3]
        self.brackets = [i for i, x in enumerate(
            self.tokens) if x == "{" or x == "}"]
        if len(self.brackets) == 4:
            self._nfgPayoff()
        else:
            self._nfgOutcome()
        self.game['sum_shape'] = sum(self.game['shape'])
        self.game['array'] = []
        for player in xrange(self.game['num_players']):
            self.game['array'].append(np.ndarray(
                self.game['shape'], dtype=float, order="F"))
        it = np.nditer(self.game['array'][0], flags=['multi_index', 'refs_ok'])
        index = 0
        while not it.finished:
            for player in xrange(self.game['num_players']):
                self.game['array'][player][
                    it.multi_index] = self.payoffs[index][player]
            it.iternext()
            index += 1
        return self.game

    def readFile(self, file):
        """
        Read content of nfg file.

        :param file: path to file
        :type file: str
        :return: dictionary with game informations
        :rtype: dict
        """
        with open(file) as f:
            return self.readStr(f.read())

    def _nfgPayoff(self):
        """
        Reads content of tokens in nfg payoff format.
        """
        self.game['players'] = self.tokens[
            self.brackets[0] + 1:self.brackets[1]]
        self.game['num_players'] = len(self.game['players'])
        self.game['shape'] = self.tokens[self.brackets[2] + 1:self.brackets[3]]
        self.game['shape'] = map(int, self.game['shape'])
        payoffs_flat = self.tokens[self.brackets[3] + 1:self.brackets[3] + 1 +
                                   reduce(mul, self.game['shape']) * self.game['num_players']]
        payoffs_flat = map(float, payoffs_flat)
        self.payoffs = []
        for i in xrange(0, len(payoffs_flat), self.game['num_players']):
            self.payoffs.append(payoffs_flat[i:i + self.game['num_players']])

    def _nfgOutcome(self):
        """
        Reads content of tokens in nfg outcome format.
        """
        brackets_pairs = []
        for i in self.brackets:
            if self.tokens[i] == "{":
                brackets_pairs.append([i])
            if self.tokens[i] == "}":
                pair = -1
                while len(brackets_pairs[pair]) != 1:
                    pair -= 1
                brackets_pairs[pair].append(i)
        self.game['players'] = self.tokens[
            self.brackets[0] + 1:self.brackets[1]]
        self.game['num_players'] = len(self.game['players'])
        i = 2
        self.game['shape'] = []
        while brackets_pairs[i][1] < brackets_pairs[1][1]:
            self.game['shape'].append(brackets_pairs[
                                      i][1] - brackets_pairs[i][0] - 1)
            i += 1
        after_brackets = brackets_pairs[i][1] + 1
        i += 1
        outcomes = [[0] * self.game['num_players']]
        for i in xrange(i, len(brackets_pairs)):
            outcomes.append(
                map(lambda x: float(x.translate(None, ',')),
                    self.tokens[brackets_pairs[i][0] + 2:brackets_pairs[i][1]]))
        self.payoffs = [outcomes[out]
                        for out in map(int, self.tokens[after_brackets:])]


def read(content):
    gr = GameReader()
    if os.path.isfile(content):
        return gr.readFile(content)
    else:
        return gr.readStr(content)
