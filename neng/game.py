#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import shlex
from operator import mul
import scipy.optimize
import sys
import logging
import cmaes
import support_enumeration


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

        Args:
            nfg (str) string containing the game in nfg format
            trim ('normalization'|'penalization') method of assuring that
                strategy profile lies in Delta
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

        Args:
            player (int) player who should respond
            strategy () opponent strategy

        Returns:
            set of best response strategies
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

        Returns:
            set of strategy profiles that was computed as pure nash equilibria
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
        result = map(self.coordinateToStrategyProfile, ne_coordinates)
        return result

    def getDominatedStrategies(self):
        """
        Returns:
            list of dominated strategies per player
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
                    self.deleted_strategies[player] = np.append(self.deleted_strategies[player], original_strategy)#strategy + np.sum(self.deleted_strategies[player] <= strategy))
                self.shape[player] -= len(strategies)
            self.sum_shape = sum(self.shape)
            dominated_strategies = self.getDominatedStrategies()
        for player in xrange(self.num_players):
            self.deleted_strategies[player].sort()

    def isDegenerate(self):
        """
        Degenerate game is defined for two-players games and there can be
        infinite number of mixed Nash equilibria.

        Returns:
            True|False if game is said as degenerated
        """
        if self.num_players != 2:
            return False
        if self.brs is None:
            self.getPNE()
        num_brs = [len(x) for x in self.brs]
        num_strategies = [reduce(mul, self.shape[:k] + self.shape[(k+1):]) for k in xrange(self.num_players)]
        if num_brs != num_strategies:
            return True
        else:
            return False

    def LyapunovFunction(self, strategy_profile):
        """
        Lyapunov function. If LyapunovFunction(p) == 0 then p is NE.

        xij(p) = ui(si, p_i)
        yij(p) = xij(p) - ui(p)
        zij(p) = max[yij(p), 0]
        LyapunovFunction(p) = sum_{i \in N} sum_{1 <= j <= mi} [zij(p)]^2

        Beside this function we need that strategy_profile is in universum
        Delta (basicaly to have character of probabilities for each player).
        We can assure this with two methods: normalization and penalization.

        Args:
            strategy_profile list of parameters to function

        Returns:
            value of LyapunovFunction in given strategy_profile
        """
        v = 0.0
        acc = 0
        deep_strategy_profile = self.strategyProfileToDeep(strategy_profile)
        if self.deltaAssuranceMethod == 'normalization':
            deep_strategy_profile = self.normalizeDeepStrategyProfile(deep_strategy_profile)
        else:
            strategy_profile_repaired = np.clip(strategy_profile, 0, 1)
            out_of_box_penalty = np.sum((strategy_profile - strategy_profile_repaired) ** 2)
            v += out_of_box_penalty
        for player in range(self.num_players):
            u = self.payoff(deep_strategy_profile, player)
            if self.deltaAssuranceMethod == 'penalization':
                one_sum_penalty = (1 - np.sum(strategy_profile[acc:acc+self.shape[player]])) ** 2
                v += one_sum_penalty
            acc += self.shape[player]
            for pure_strategy in range(self.shape[player]):
                x = self.payoff(deep_strategy_profile, player, pure_strategy)
                z = x - u
                g = max(z, 0.0)
                v += g ** 2
        return v
    
    def payoff(self, strategy_profile, player, pure_strategy=None):
        """
        Function to compute payoff of given strategy_profile.

        Args:
            strategy_profile list of probability distributions
            player player for who the payoff is computated
            pure_strategy if not None player strategy will be replaced by pure_strategy

        Returns:
            value of payoff
        """
        result = 0.0
        if len(strategy_profile) == self.num_players:
            deep_strategy_profile = strategy_profile[:]
            if pure_strategy is not None:
                new_strategy = np.zeros_like(deep_strategy_profile[player])
                new_strategy[pure_strategy] = 1.0
                deep_strategy_profile[player] = new_strategy
        elif len(strategy_profile) == self.sum_shape:
            deep_strategy_profile = self.strategyProfileToDeep(strategy_profile)
        else:
            raise Exception("Length of strategy_profile: '{0}', does not match.".format(strategy_profile))
        # make product of each probability, returns num_players-dimensional array 
        product = reduce(lambda x, y: np.tensordot(x, y, 0), deep_strategy_profile)
        result = np.sum(product * self.array[player])
        return result

    def strategyProfileToDeep(self, strategy_profile):
        """
        Convert strategy_profile to deep_strategy_profile.
        It means that instead of list of length sum_shape we have got nested
        list of length num_players and inner arrays are of shape[player] length

        Args:
            strategy_profile to convert

        Returns:
            deep_strategy_profile
        """
        offset = 0
        deep_strategy_profile = []
        for player, i in enumerate(self.shape):
            strategy = strategy_profile[offset:offset+i]
            deep_strategy_profile.append(strategy)
            offset += i
        return deep_strategy_profile

    def normalize(self, strategy):
        """
        Normalize strategy_profile to asure constraints:
        for all strategies sum p(si) = 1
        p(si) >= 0.0

        Args:
            strategy np.array of probability distribution for one player

        Returns:
            np.array normalized strategy distribution
        """
        return np.abs(strategy) / np.sum(np.abs(strategy))

    def normalizeDeepStrategyProfile(self, deep_strategy_profile):
        for index, strategy in enumerate(deep_strategy_profile):
            deep_strategy_profile[index] = self.normalize(strategy)
        return deep_strategy_profile


    def normalizeStrategyProfile(self, strategy_profile):
        """
        Normalize whole strategy profile by strategy of each player

        Args:
            strategy_profile to be normalized
        
        Returns:
            normalized strategy_profile
        """
        result = []
        acc = 0
        for i in self.shape:
            strategy = np.array(strategy_profile[acc:acc+i])
            result.extend(self.normalize(strategy))
            acc += i
        return result

    def findEquilibria(self, method='CMAES'):
        """
        Find all equilibria, using method

        Args:
            method method from Game.METHODS to be used

        Returns:
            list of NE(list of probabilities), if not found return None
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
                                             options={"maxiter":1e3 * self.sum_shape ** 2})
        logging.info(result)
        if result.success:
            r = []
            r.append(result.x)
            return r
        else:
            return None

    def read(self, nfg):
        """
        Reads game in .nfg format and stores data to class variables.
        Can read nfg files in outcome and payoff version.

        Args:
            nfg string with nfg formated game
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
            outcomes = []
            outcomes.append([0] * self.num_players)
            for i in xrange(i, len(brackets_pairs)):
                outcomes.append(map(lambda x: float(x.translate(None, ',')), tokens[brackets_pairs[i][0] + 2:brackets_pairs[i][1]]))
            payoffs = [outcomes[out] for out in map(int, tokens[after_brackets:])]
        self.sum_shape = np.sum(self.shape)
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

        Returns:
            game in nfg payoff format
        """
        result = "NFG 1 R "
        result += "\"" + self.name + "\"\n"
        result += "{ "
        result += " ".join(map(lambda x: "\"" + x + "\"", self.players))
        result += " } { "
        result += " ".join(map(str, self.shape))
        result += " }\n\n"
        for payoff in np.nditer(self.array[0], order="F", flags=['refs_ok']):
            for i in payoff.flat[0]:
                    result += str(i) + " "
        return result

    def coordinateToStrategyProfile(self, t):
        """
        Translate tuple form of strategy profile to list of probabilities

        Args:
            t tuple to translate

        Returns:
            list of numbers in long format
        """
        result = [0.0] * self.sum_shape
        offset = 0
        for index, i in enumerate(self.shape):
            result[t[index] + offset] = 1.0
            offset += i
        return result

    def printNE(self, nes, payoff=False, checkNE=False):
        """
        Print Nash equilibria with with some statistics

        Args:
            nes list of nash equilibria
            payoff bool flag to print payoff also
            checkNE bool run test for every printed NE
        """
        result = ""
        success = True
        if self.degenerate:
            logging.warning("Game is degenerated")
        for index, ne in enumerate(nes):
            ne = self.normalizeStrategyProfile(ne)
            print_ne = list(ne)
            # assure that printed result are in same shape as self.init_shape
            if self.deleted_strategies is not None:
                acc = 0
                for player in xrange(self.num_players):
                    for deleted_strategy in self.deleted_strategies[player]:
                        print_ne.insert(acc + deleted_strategy, 0.0)
                    acc += self.init_shape[player]
            probabilities = map(str, print_ne)
            result += "NE " + ", ".join(probabilities)
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

        Args:
            strategy_profile check if is NE

        Returns:
            True if strategy_profile passed test, False otherwise
        """
        payoffs = []
        deep_strategy_profile = []
        acc = 0
        for player, i in enumerate(self.shape):
            strategy = np.array(strategy_profile[acc:acc+i])
            deep_strategy_profile.append(strategy)
            acc += i
            #payoffs
            payoffs.append(self.payoff(strategy_profile, player))
        for player in xrange(self.num_players):
            dsp = deep_strategy_profile[:]
            empty_strategy = [0.0] * self.shape[player]
            for strategy in xrange(self.shape[player]):
                es = empty_strategy[:]
                es[strategy] = 1.0
                dsp[player] = es
                current_payoff = self.payoff(dsp, player)
                if (current_payoff - payoffs[player]) > accuracy:
                    logging.warning('Player {0} has better payoff with {1}, previous payoff {2}, current payoff {3}, difference {4}. '.format(player, dsp[player], payoffs[player],
                                    current_payoff, payoffs[player] - current_payoff))
                    logging.warning("NE test failed")
                    return False
            for i in xrange(num_tests):
                dsp[player] = self.normalize(np.random.rand(self.shape[player]))
                current_payoff = self.payoff(dsp, player)
                if (current_payoff - payoffs[player]) > accuracy:
                    logging.warning('Player {0} has better payoff with {1}, previous payoff {2}, current payoff {3}, difference {4}. '.format(player, dsp[player], payoffs[player],
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
    parser.add_argument('-m', '--method', default='CMAES', choices=Game.METHODS, help="Method to use for computing Nash equlibria.")
    parser.add_argument('-e', '--elimination', action='store_true', default=False, help="Use Iterative Elimination of Strictly Dominated Strategies before computing NE.")
    parser.add_argument('-p', '--payoff', action='store_true', default=False, help="Print also players payoff with each Nash equilibrium.")
    parser.add_argument('-c', '--checkNE', action='store_true', default=False, help="After computation check if found strategy profile is really Nash equilibrium.")
    parser.add_argument('-t', '--trim', choices=('normalization', 'penalization'), default='normalization', help="Method for keeping strategy profile in probability distribution universum.")
    parser.add_argument('-l', '--log', default="WARNING", choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"), help="Level of logs to save/print")
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
        text, success =  g.printNE(result, payoff=args.payoff, checkNE=args.checkNE)
        if success:
            print text
        else:
            sys.exit("Nash equilibrium did not pass the test.")
    else:
        sys.exit("Nash equilibrium was not found.")
