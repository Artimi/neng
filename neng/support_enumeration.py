#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging
import itertools

class SupportEnumeration(object):
    """
    Class providing support enumeration method for finding all mixed Nash
    equilibria in two-players games.
    """

    def __init__(self, game):
        self.game = game

    def getEquationSet(self, combination, player, num_supports):
        """
        Return set of equations for given player and combination of strategies
        for 2 players games in support_enumeration

        This function returns matrix to compute (Nisan algorithm 3.4)
        (I = combination[player])
        \sum_{i \in I} x_i b_{ij} = v
        \sum_{i \in I} x_i = 1
        In matrix form (k = num_supports):
        / b_11 b_12 ... b_1k -1 \ / x_1 \    / 0 \
        | b_21 b_22 ... b_2k -1 | | x_2 |    | 0 |
        | ...  ...  ... ... ... | | ... | =  |...|
        | b_k1 b_k2 ... b_kk -1 | | x_k |    | 0 |
        \ 1    1    ... 1     0 / \ v   /    \ 1 /

        Args:
            combination combination of strategies to make equation set
            player number of player for who the equation matrix will be done
            num_supports number of supports for players

        Returns:
            equation matrix for solving in e.g. np.linalg.solve
        """
        row_index = np.zeros(self.game.shape[0], dtype=bool)
        col_index = np.zeros(self.game.shape[1], dtype=bool)
        row_index[list(combination[0])] = True
        col_index[list(combination[1])] = True
        numbers = self.game.array[(player + 1) % 2][row_index][:, col_index]
        last_row = np.ones((1, num_supports + 1))
        last_row[0][-1] = 0
        last_column = np.ones((num_supports, 1)) * -1
        if player == 0:
            numbers = numbers.T
        numbers = np.hstack((numbers, last_column))
        numbers = np.vstack((numbers, last_row))
        return numbers

    def supportEnumeration(self):
        """
        Computes all mixed NE of 2 player noncooperative games.
        If the game is degenerate game.degenerate flag is ticked.

        Returns:
            set of NE computed by method support enumeration
        """
        result = []
        # for every numbers of supports
        for num_supports in xrange(1, min(self.game.shape) + 1):
            logging.debug("Support enumearation for num_supports: {0}".format(num_supports))
            supports = []
            equal = [0] * num_supports
            equal.append(1)
            # all combinations of support length num_supports
            for player in xrange(self.game.num_players):
                supports.append(itertools.combinations(
                    xrange(self.game.shape[player]), num_supports))
            # cartesian product of combinations of both player
            for combination in itertools.product(supports[0], supports[1]):
                mne = []
                is_mne = True
                # for both player compute set of equations
                for player in xrange(self.game.num_players):
                    equations = self.getEquationSet(combination, player,
                                                      num_supports)
                    try:
                        equations_result = np.linalg.solve(equations, equal)
                    except np.linalg.LinAlgError:  # unsolvable equations
                        is_mne = False
                        break
                    probabilities = equations_result[:-1]
                    # all probabilities are nonnegative
                    if not np.all(probabilities >= 0):
                        is_mne = False
                        break
                    player_strategy_profile = np.zeros(self.game.shape[player])
                    player_strategy_profile[list(combination[player])] = probabilities
                    mne.append(player_strategy_profile)
                #best response
                if is_mne:
                    br = [[],[]]
                    for player in xrange(self.game.num_players):
                        if player == 0:
                            oponent_strategy = mne[(player + 1) % 2].reshape(1, self.game.shape[1])
                        else:
                            oponent_strategy = mne[(player + 1) % 2].reshape(self.game.shape[0], 1)
                        payoffs = np.sum(self.game.array[player] * oponent_strategy, axis=(player + 1) % 2)
                        maximum = np.max(payoffs)
                        br[player] = tuple(np.where(abs(payoffs - maximum)< 1e-6)[0])
                        if br[player] != combination[player]:
                            is_mne = False
                            break
                    #if len(br[0]) != len(br[1]):
                        #self.game.degenerate = True
                if is_mne:
                    result.append([item for sublist in mne for item in sublist])
        return result

def computeNE(game):
    """
    Function for easy calling SupportEnumeration from other modules.
    """
    se = SupportEnumeration(game)
    return se.supportEnumeration()
