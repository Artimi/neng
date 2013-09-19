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

from __future__ import unicode_literals


class ExtensiveTree(object):
    """
    This class hold general information about game in extensive form.
    """
    def __init__(self, name, players, root=None):
        """
        :param name: name of the extensive game
        :type name: str
        :param players: player names
        :type players: list
        :param root: root element of created tree
        :type root: neng.ExtensiveTreeNode
        """
        self.name = name
        self.players = players
        self.root = root

    def __repr__(self):
        return "{}: [{}]".format(self.name, ", ".join(self.players))


class ExtensiveTreeNode(object):
    """
    ExtensiveTreeNode represents node in the game. Currently it is used for
    terminal nodes.
    """
    def __init__(self, name, outcome, payoffs, outcome_name=""):
        """
        :param name: node name
        :type name: str
        :param outcome: a nonnegative integer specifying the outcome
        :type outcome: int
        :param payoffs: list of payoffs
        :type payoffs: list
        :param outcome_name: outcome name
        :type outcome_name: str
        """
        self.name = name
        self.outcome = outcome
        self.outcome_name = outcome_name
        self.payoffs = payoffs
        self.children = []
        self.parent = None

    def add_child(self, child):
        """
        Add child to this node.

        :param child: child to be added
        :type child: neng.ExtensiveTreeNode
        """
        self.children.append(child)
        child.parent = self

    def __contains__(self, child):
        """
        Method to test if node has child

        :param child: child to test if is in the node
        :type child: neng.ExtensiveTreeNode
        """
        return child in self.children

    def __repr__(self):
        return "{}: {}#{}, {}".format(self.name, self.outcome_name,
                                      self.outcome, self.payoffs)


class ExtensiveTreePersonNode(ExtensiveTreeNode):
    """
    ExtensiveTreePersonNode represent node for persons decisions.
    """
    def __init__(self, name, outcome, payoffs, player, information_set,
                 outcome_name="", information_set_name="", action_names=[]):
        """
        :param name: node name
        :type name: str
        :param outcome: a nonnegative integer specifying the outcome
        :type outcome: int
        :param payoffs: list of payoffs
        :type payoffs: list
        :param player: player that owns node
        :type player: int
        :param information_set: integer specifying the information set
        :type information_set: int
        :param outcome_name: outcome name
        :type outcome_name: str
        :param information_set_name: name of information_set
        :type information_set_name: str
        :param action_names: names of action player can perform
        :type action_names: list
        """
        self.player = player
        self.information_set = information_set
        self.information_set_name = information_set_name
        self.action_names = action_names
        super(ExtensiveTreePersonNode, self).__init__(name, outcome,
                                                      outcome_name, payoffs)

    def __repr__(self):
        return "{}: player: {}, information_set: {}".format(self.name,
                                                            self.player,
                                                            self.information_set)
