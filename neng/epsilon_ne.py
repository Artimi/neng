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

import strategy_profile as sp


class EpsilonNE(object):
    """
    Class containing epsilon-approximate Nash equilibrium algorithms
    """

    def __init__(self, game):
        self.game = game

    def compute(self, quotient):
        quotients = {'0.5': self._05}
        if quotient in quotients:
            return quotients[quotient]()
        else:
            raise Exception("Unknown epsilon quotient {0}.".format(quotient))

    def _05(self):
        result = set()
        for player in xrange(self.game.num_players):
            opponent = (player + 1) % 2
            for i in xrange(self.game.shape[player]):
                brs = self.game.pureBestResponseTwoPlayers(opponent, i)
                j = brs.pop()[opponent]
                brs = self.game.pureBestResponseTwoPlayers(player, j)
                k = brs.pop()[player]
                profile = sp.StrategyProfile(None, self.game.shape)
                profile.updateWithList(player, [(i, 0.5), (k, 0.5)])
                profile.updateWithList(opponent, [(j, 1.0)])
                result.add(profile)
        return list(result)


def compute(game, quotient):
    """
    Function for easy call EpsilonNE from other modules.
    """
    ene = EpsilonNE(game)
    return ene.compute(quotient)
