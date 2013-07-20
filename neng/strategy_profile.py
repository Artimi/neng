#!/usr/bin/env python
#-*- coding: UTF-8 -*-

#Copyright (C) 2013 Petr Šebek

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in
#all copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHERWISE
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

from __future__ import division
import copy


class StrategyProfile(object):
    """
    Wraps information about strategy profile of game.
    """

    def __init__(self, profile, shape, coordinate=False):
        self.shape = shape
        self._list = []
        if coordinate:
            self._coordinateToDeep(profile)
        else:
            self._flatToDeep(profile)

    def _coordinateToDeep(self, coordinate):
        for player in xrange(len(self.shape)):
            self._list.append([])
            self.updateWithPureStrategy(player, coordinate[player])

    def _flatToDeep(self, profile):
        """
        Convert strategy_profile to deep strategy profile.
        It means that instead of list of length sum_shape we have got nested
        list of length num_players and inner arrays are of shape[player] length

        Args:
            strategy_profile to convert

        Returns:
            deep strategy profile
        """
        offset = 0
        for player, i in enumerate(self.shape):
            strategy = profile[offset:offset + i]
            self._list.append(strategy)
            offset += i
        return self

    def normalize(self):
        for player, strategy in enumerate(self._list):
            sumation = 0
            for i in strategy:
                sumation += abs(i)
            for index, value in enumerate(strategy):
                self._list[player][index] = abs(value) / sumation
        return self

    def copy(self):
        return copy.deepcopy(self)

    def updateWithPureStrategy(self, player, strategy):
        self._list[player] = [0.0] * self.shape[player]
        self._list[player][strategy] = 1.0

    def __str__(self):
        result = ''
        flat_profile = [item for sublist in self._list for item in sublist]
        result += ', '.join(map(str, flat_profile))
        return result

    def __repr__(self):
        return self._list.__repr__()

    def __setitem__(self, key, value):
        if self.shape[key] != len(value):
            raise IndexError("Strategy has to be same length as corresponding shape value is.")
        else:
            self._list.__setitem__(key, value)

    def __getitem__(self, item):
        return self._list.__getitem__(item)

    def __eq__(self, other):
        return self._list == other._list
