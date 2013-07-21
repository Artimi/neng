#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

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
import shlex
from operator import mul
import sys
import logging

import numpy as np
import scipy.optimize

import cmaes
import support_enumeration
import strategy_profile as sp


class Game(object):
    """
    Class Game wrap around all informations of noncooperative game. Also
    it provides basic analyzation of game, like bestResponse, if the game
    is degenerate. It also contains an algorithm for iterative elimination
    of strictly dominated strategies and can compute pure Nash equilibria
    using brute force.

    usage:
        >>> g = Game(game_str)
        >>> ne = g.findEquilibria(method='pne')
        >>> print g.printNE(ne)
    """
    METHODS = ['L-BFGS-B', 'SLSQP', 'CMAES', 'support_enumeration', 'pne']

    def __init__(self, nfg, trim='normalization'):
        """
        Initialize basic attributes in Game

        :param nfg: string containing the game in nfg format
        :type nfg: str
        :param trim: method of assuring that strategy profile lies in Delta space,'normalization'|'penalization'
        :type trim: str
        """
        self.read(nfg)
        self.deltaAssuranceMethod = trim
        self.players_zeros = np.zeros(self.num_players)
        self.brs = None
        self.degenerate = None
        self.deleted_strategies = None

    def bestResponse(self, player, strategy):
        """
        Computes pure best response strategy profile for given opponent strategy 
        and player

        :param player: player who should respond
        :type player: int
        :param strategy: opponnet strategy
        :type strategy: list
        :return: set of best response strategies
        :rtype: set of coordinates
        """
        strategy = list(strategy)
        result = set()
        strategy[player] = slice(None)  # all possible strategies for 'player'
        payoffs = self.array[player][strategy]
        max_payoff = np.max(payoffs)
        # numbers of best responses strategies
        brs = [index for index, br in enumerate(payoffs) if br == max_payoff]
        for br in brs:
            s = strategy[:]
            s[player] = br
            # made whole strategy profile, not just one strategy
            result.add(tuple(s))
        return result

    def getPNE(self):
        """
        Function computes pure Nash equlibria using brute force algorithm.

        :return: list of StrategyProfile that are pure Nash equilibria
        :rtype: list
        """
        self.brs = [set() for i in xrange(self.num_players)]
        for player in xrange(self.num_players):
            p_view = self.shape[:]
            p_view[player] = 1
            # get all possible opponent strategy profiles to 'player'
            for strategy in np.ndindex(*p_view):
                # add to list of best responses
                self.brs[player].update(self.bestResponse(player, strategy))
                # check degeneration of a game
        self.degenerate = self.isDegenerate()
        # PNE is where all player have best response
        ne_coordinates = set.intersection(*self.brs)
        result = map(lambda coordinate: sp.StrategyProfile(coordinate, self.shape, coordinate=True), ne_coordinates)
        return result

    def getDominatedStrategies(self):
        """
        :return: list of dominated strategies per player
        :rtype: list
        """
        empty = [slice(None)] * self.num_players
        result = []
        for player in xrange(self.num_players):
            s1 = empty[:]
            strategies = []
            dominated_strategies = []
            for strategy in xrange(self.shape[player]):
                s1[player] = strategy
                strategies.append(self.array[player][s1])
            for strategy in xrange(self.shape[player]):
                dominated = False
                for strategy2 in xrange(self.shape[player]):
                    if strategy == strategy2:
                        continue
                    elif (strategies[strategy] < strategies[strategy2]).all():
                        dominated = True
                        break
                if dominated:
                    dominated_strategies.append(strategy)
            result.append(dominated_strategies)
        return result

    def IESDS(self):
        """
        Iterative elimination of strictly dominated strategies.

        Eliminates all strict dominated strategies, preserve self.array and
        self.shape in self.init_array and self.init_shape. Stores numbers of
        deleted strategies in self.deleted_strategies. Deletes strategies
        from self.array and updates self.shape.
        """
        self.init_array = self.array[:]
        self.init_shape = self.shape[:]
        self.deleted_strategies = [np.array([], dtype=int) for player in xrange(self.num_players)]
        dominated_strategies = self.getDominatedStrategies()
        while sum(map(len, dominated_strategies)) != 0:
            logging.debug("Dominated strategies to delete: {0}".format(dominated_strategies))
            for player, strategies in enumerate(dominated_strategies):
                for p in xrange(self.num_players):
                    self.array[p] = np.delete(self.array[p], strategies, player)
                for strategy in strategies:
                    original_strategy = strategy
                    while original_strategy in self.deleted_strategies[player]:
                        original_strategy += 1
                    self.deleted_strategies[player] = np.append(self.deleted_strategies[player],
                                                                original_strategy)
                self.shape[player] -= len(strategies)
            self.sum_shape = sum(self.shape)
            dominated_strategies = self.getDominatedStrategies()
        for player in xrange(self.num_players):
            self.deleted_strategies[player].sort()

    def isDegenerate(self):
        """
        Degenerate game is defined for two-players games and there can be
        infinite number of mixed Nash equilibria.

        :return: True if game is said as degenerated
        :rtype: bool
        """
        if self.num_players != 2:
            return False
        if self.brs is None:
            self.getPNE()
        num_brs = [len(x) for x in self.brs]
        num_strategies = [reduce(mul, self.shape[:k] + self.shape[(k + 1):]) for k in xrange(self.num_players)]
        if num_brs != num_strategies:
            return True
        else:
            return False

    def LyapunovFunction(self, strategy_profile_flat):
        r"""
        Lyapunov function. If LyapunovFunction(p) == 0 then p is NE.
        
        .. math::

            x_{ij}(p)           & = u_{i}(si, p_i) \\
            y_{ij}(p)           & = x_{ij}(p) - u_i(p) \\
            z_{ij}(p)           & = \max[y_{ij}(p), 0] \\
            LyapunovFunction(p) & = \sum_{i \in N} \sum_{1 \leq j \leq \mu} z_{ij}(p)^2

        Beside this function we need that strategy_profile is in universum
        Delta (basicaly to have character of probabilities for each player).
        We can assure this with two methods: normalization and penalization.

        :param strategy_profile_flat: list of parameters to function
        :type strategy_profile_flat: list
        :return: value of Lyapunov function in given strategy profile
        :rtype: float
        """
        v = 0.0
        acc = 0
        strategy_profile = sp.StrategyProfile(strategy_profile_flat, self.shape)
        if self.deltaAssuranceMethod == 'normalization':
            strategy_profile.normalize()
        else:
            strategy_profile_repaired = np.clip(strategy_profile_flat, 0, 1)
            out_of_box_penalty = np.sum((strategy_profile_flat - strategy_profile_repaired) ** 2)
            v += out_of_box_penalty
        for player in range(self.num_players):
            u = self.payoff(strategy_profile, player)
            if self.deltaAssuranceMethod == 'penalization':
                one_sum_penalty = (1 - np.sum(strategy_profile[player])) ** 2
                v += one_sum_penalty
            acc += self.shape[player]
            for pure_strategy in range(self.shape[player]):
                x = self.payoff(strategy_profile, player, pure_strategy)
                z = x - u
                g = max(z, 0.0)
                v += g ** 2
        return v

    def payoff(self, strategy_profile, player, pure_strategy=None):
        """
        Function to compute payoff of given strategy_profile.

        :param strategy_profile: strategy profile of all players
        :type strategy_profile: StrategyProfile
        :param player: player for whom the payoff is computed
        :type player: int
        :param pure_strategy: if not None player strategy will be replaced by pure strategy of that number
        :type pure_strategy: int
        :return: value of payoff
        :rtype: float
        """
        sp = strategy_profile.copy()
        if pure_strategy is not None:
            sp.updateWithPureStrategy(player, pure_strategy)
        # make product of each probability, returns num_players-dimensional array
        product = reduce(lambda x, y: np.tensordot(x, y, 0), sp)
        result = np.sum(product * self.array[player])
        return result

    def findEquilibria(self, method='CMAES'):
        """
        Find all equilibria, using method

        :param method: of computing equilibria
        :type method: str, one of Game.METHODS
        :return: list of NE, if not found returns None
        :rtype: list of StrategyProfile
        """
        if method == 'pne':
            result = self.getPNE()
            if len(result) == 0:
                return None
            else:
                return result
        elif self.num_players == 2 and method == 'support_enumeration':
            result = support_enumeration.computeNE(self)
            self.degenerate = self.isDegenerate()
            if len(result) == 0:
                return None
            else:
                return result
        elif method == 'CMAES':
            result = cmaes.fmin(self.LyapunovFunction, self.sum_shape)
        elif method in self.METHODS:
            result = scipy.optimize.minimize(self.LyapunovFunction,
                                             np.random.rand(self.sum_shape),
                                             method=method, tol=1e-10,
                                             options={"maxiter": 1e3 * self.sum_shape ** 2})
        logging.info(result)
        if result.success:
            r = [sp.StrategyProfile(result.x, self.shape)]
            return r
        else:
            return None

    def read(self, nfg):
        """
        Reads game in .nfg format and stores data to class variables.
        Can read nfg files in outcome and payoff version.

        :param nfg: nfg formated game
        :type nfg: str
        """
        tokens = shlex.split(nfg)
        preface = ["NFG", "1", "R"]
        if tokens[:3] != preface:
            raise Exception("Input string is not valid nfg format")
        self.name = tokens[3]
        brackets = [i for i, x in enumerate(tokens) if x == "{" or x == "}"]
        if len(brackets) == 4:
            # payoff version
            self.players = tokens[brackets[0] + 1:brackets[1]]
            self.num_players = len(self.players)
            self.shape = tokens[brackets[2] + 1:brackets[3]]
            self.shape = map(int, self.shape)
            payoffs_flat = tokens[brackets[3] + 1:brackets[3] + 1 +
                                                  reduce(mul, self.shape) * self.num_players]
            payoffs_flat = map(float, payoffs_flat)
            payoffs = []
            for i in xrange(0, len(payoffs_flat), self.num_players):
                payoffs.append(payoffs_flat[i:i + self.num_players])
        else:
            # outcome verion
            brackets_pairs = []
            for i in brackets:
                if tokens[i] == "{":
                    brackets_pairs.append([i])
                if tokens[i] == "}":
                    pair = -1
                    while len(brackets_pairs[pair]) != 1:
                        pair -= 1
                    brackets_pairs[pair].append(i)
            self.players = tokens[brackets[0] + 1:brackets[1]]
            self.num_players = len(self.players)
            i = 2
            self.shape = []
            while brackets_pairs[i][1] < brackets_pairs[1][1]:
                self.shape.append(brackets_pairs[i][1] - brackets_pairs[i][0] - 1)
                i += 1
            after_brackets = brackets_pairs[i][1] + 1
            i += 1
            outcomes = [[0] * self.num_players]
            for i in xrange(i, len(brackets_pairs)):
                outcomes.append(
                    map(lambda x: float(x.translate(None, ',')), tokens[brackets_pairs[i][0] + 2:brackets_pairs[i][1]]))
            payoffs = [outcomes[out] for out in map(int, tokens[after_brackets:])]
        self.sum_shape = sum(self.shape)
        self.array = []
        for player in xrange(self.num_players):
            self.array.append(np.ndarray(self.shape, dtype=float, order="F"))
        it = np.nditer(self.array[0], flags=['multi_index', 'refs_ok'])
        index = 0
        while not it.finished:
            for player in xrange(self.num_players):
                self.array[player][it.multi_index] = payoffs[index][player]
            it.iternext()
            index += 1

    def __str__(self):
        """
        Output in nfg payoff format.

        :return: game in nfg payoff format
        :rtype: str
        """
        result = "NFG 1 R "
        result += "\"" + self.name + "\"\n"
        result += "{ "
        result += " ".join(map(lambda x: "\"" + x + "\"", self.players))
        result += " } { "
        result += " ".join(map(str, self.shape))
        result += " }\n\n"
        it = np.nditer(self.array[0], order='F', flags=['multi_index', 'refs_ok'])
        payoffs = []
        while not it.finished:
            for player in xrange(self.num_players):
                payoffs.append(self.array[player][it.multi_index])
            it.iternext()
        result += " ".join(map(str, payoffs))
        return result

    def printNE(self, nes, payoff=False, checkNE=False):
        """
        Print Nash equilibria with with some statistics

        :param nes: list of Nash equilibria
        :type nes: list of StrategyProfile
        :param payoff: flag to print payoff of each player
        :type payoff: bool
        :param checkNE: run test for every printed NE
        :type checkNE: bool
        :return: string to print and information about checking NE
        :rtype: tuple, (str, bool)
        """
        result = ""
        success = True
        if self.degenerate:
            logging.warning("Game is degenerated")
        for index, ne in enumerate(nes):
            ne.normalize()
            # assure that printed result are in same shape as self.init_shape
            if self.deleted_strategies is not None:
                for player in xrange(self.num_players):
                    for deleted_strategy in self.deleted_strategies[player]:
                        ne[player].insert(deleted_strategy, 0.0)
                        ne._shape[player] += 1
            result += "NE " + str(ne)
            if index != len(nes) - 1:
                result += '\n'
            if payoff:
                s = []
                for player in xrange(self.num_players):
                    s.append("{0}: {1:.3f}".format(self.players[player], self.payoff(ne, player)))
                result += "Payoff " + ", ".join(s) + "\n"
            if checkNE:
                if not self.checkNE(ne):
                    success = False
        return result, success

    def checkNE(self, strategy_profile, num_tests=1000, accuracy=1e-4):
        """
        Function generates random probability distribution for players and
        check if strategy_profile is really NE. If the payoff will be bigger
        it's not the NE.

        :param strategy_profile: strategy profile to check
        :type strategy_profile: StrategyProfile
        :param num_tests: count of tests to do
        :type num_tests: int
        :param accuracy: accuracy of possible difference of results
        :type accuracy: float
        :return: True if strategy_profile passed test, False otherwise
        :rtype: bool
        """
        payoffs = []
        for player in xrange(self.num_players):
            payoffs.append(self.payoff(strategy_profile, player))
        for player in xrange(self.num_players):
            sp = strategy_profile.copy()
            for strategy in xrange(self.shape[player]):
                sp.updateWithPureStrategy(player, strategy)
                current_payoff = self.payoff(sp, player)
                if (current_payoff - payoffs[player]) > accuracy:
                    logging.warning(
                        'Player {0} has better payoff with {1}, previous payoff {2}, current payoff {3}, difference {4}. '.format(
                            player, sp[player], payoffs[player],
                            current_payoff, payoffs[player] - current_payoff))
                    logging.warning("NE test failed")
                    return False
            for i in xrange(num_tests):
                sp[player] = list(np.random.rand(self.shape[player]))
                sp.normalize()
                current_payoff = self.payoff(sp, player)
                if (current_payoff - payoffs[player]) > accuracy:
                    logging.warning(
                        'Player {0} has better payoff with {1}, previous payoff {2}, current payoff {3}, difference {4}. '.format(
                            player, sp[player], payoffs[player],
                            current_payoff, payoffs[player] - current_payoff))
                    logging.warning("NE test failed")
                    return False
        logging.info("NE test passed")
        return True


if __name__ == '__main__':
    import argparse
    import time

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
NenG - Nash Equilibrium Noncooperative Games.
Tool for computing Nash equilibria in noncooperative games.
Specifically:
All pure Nash equilibria in all games (--method=pne).
All mixed Nash equilibria in two-players games (--method=support_enumeration).
One sample mixed Nash equilibria in n-players games (--method={CMAES,L-BFGS-B,SLSQP}).
""")
    parser.add_argument('-f', '--file', required=True, help="File where game in nfg format is saved.")
    parser.add_argument('-m', '--method', default='CMAES', choices=Game.METHODS,
                        help="Method to use for computing Nash equlibria.")
    parser.add_argument('-e', '--elimination', action='store_true', default=False,
                        help="Use Iterative Elimination of Strictly Dominated Strategies before computing NE.")
    parser.add_argument('-p', '--payoff', action='store_true', default=False,
                        help="Print also players payoff with each Nash equilibrium.")
    parser.add_argument('-c', '--checkNE', action='store_true', default=False,
                        help="After computation check if found strategy profile is really Nash equilibrium.")
    parser.add_argument('-t', '--trim', choices=('normalization', 'penalization'), default='normalization',
                        help="Method for keeping strategy profile in probability distribution universum.")
    parser.add_argument('-l', '--log', default="WARNING", choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
                        help="Level of logs to save/print")
    parser.add_argument('--log-file', default=None, help='Log file. If omitted log is printed to stdout.')
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper(), None),
                        format="%(levelname)s, %(asctime)s, %(message)s", filename=args.log_file)

    with open(args.file) as f:
        game_str = f.read()
    start = time.time()
    g = Game(game_str, args.trim)
    logging.debug("Reading the game took: {0} s".format(time.time() - start))
    if args.elimination:
        g.IESDS()
    result = g.findEquilibria(args.method)
    if result is not None:
        text, success = g.printNE(result, payoff=args.payoff, checkNE=args.checkNE)
        if success:
            print text
        else:
            sys.exit("Nash equilibrium did not pass the test.")
    else:
        sys.exit("Nash equilibrium was not found.")
