#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Copyright (c) 2013, Petr Šebek
#All rights reserved.

#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the <organization> nor the
#      names of its contributors may be used to endorse or promote products
#      derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
#DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import division
import numpy as np
import scipy.optimize
import collections
import logging



class CMAES(object):
    """
    Class CMAES represent algorithm Covariance Matrix Adaptation - Evolution
    Strategy. It provides function minimization.
    """

    def __init__(self, func, N, sigma=0.3, xmean=None):
        """
        Args:
            func (function): function to be minimized
            N (int): number of parameter of function
            sigma (float): step size of method
            xmean (np.array): initial point, if None some is generated
        """
        self.func = func
        self.N = N
        self.store_parameters = {'func': func,
                                 'N':    N,
                                 'sigma': sigma,
                                 'xmean': xmean,
                                }
        self.stopeval = 1e4 * self.N / 2
        self.stopfitness = 1e-10
        self.eigenval = 0
        # generation loop
        self.counteval = 0
        self.generation = 0
        #stop criteria
        self.stop_criteria = ("Fitness",
                              "MaxEval",
                              "NoEffectAxis",
                              "NoEffectCoord",
                              #"Stagnation",
                              "ConditionCov"
                              "TolXUp",
                              "TolFun",
                              "TolX")
        self.tolfun = 1e-12
        self.tolxup = 1e4
        self.condition_cov_max = 1e14
        self.lamda = int(4 + 3 * np.log(self.N))
        self.initVariables(sigma, xmean)

    def initVariables(self, sigma, xmean, lamda_factor=1):
        """
        Init variables that can change after restart of method, basically that
        are dependent on lamda.

        Args:
            sigma (float): step size
            xmean (np.array): initial point
            lamda_factor (float): factor for multiplying old lamda
        """
        self.sigma = sigma
        if xmean is None:
            self.xmean = np.random.rand(self.N)
        else:
            self.xmean = xmean
        self.status = -1
        # strategy parameter setting: selection
        self.lamda *= lamda_factor
        self.mu = self.lamda / 2
        self.weights = np.array([np.log(self.mu + 0.5) - np.log(i)
                                 for i in range(1, int(self.mu) + 1)])
        self.mu = int(self.mu)
        self.weights = self.weights / np.sum(self.weights)
        self.mueff = 1 / np.sum(self.weights ** 2)
        # strategy parameter setting: adaptation
        self.cc = (4 + self.mueff / self.N) / (self.N + 4 + 2 * self.mueff / self.N)
        self.cs = (self.mueff + 2) / (self.N + self.mueff + 5)
        self.c1 = 2 / ((self.N + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) /
                       ((self.N + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) /
                                (self.N + 1)) - 1) + self.cs
        # initialize dynamic (internal) strategy parameters and constants
        self.pc = np.zeros((1, self.N))
        self.ps = np.zeros_like(self.pc)
        self.B = np.eye(self.N)
        self.D = np.eye(self.N)
        self.C = np.identity(self.N)
        self.chiN = self.N ** 0.5 * (1 - 1 / (4 * self.N) + 1 /
                                     (21 * self.N ** 2))
        # termination
        self.tolx = 1e-12 * self.sigma
        self.short_history_len = 10 + np.ceil(30 * self.N / self.lamda)
        self.long_history_len_down = 120 + 30 * self.N / self.lamda
        self.long_history_len_up = 20000
        self.history = {}
        self.history['short_best'] = collections.deque()
        self.history['long_best'] = collections.deque()
        self.history['long_median'] = collections.deque()

    def newGeneration(self):
        """
        Generate new generation of individuals.

        Returns:
            new generation
        """
        self.generation += 1
        self.arz = np.random.randn(self.lamda, self.N)
        self.arx = self.xmean + self.sigma * np.dot(np.dot(self.B, self.D), self.arz.T).T
        return self.arx

    def update(self, arfitness):
        """
        Update values of method from new evaluated generation

        Args:
            arfitness (np.array): list of func values to individuals
        """
        self.counteval += self.lamda
        self.arfitness = arfitness
        # sort by fitness and compute weighted mean into xmean
        self.arindex = np.argsort(self.arfitness)
        self.arfitness = self.arfitness[self.arindex]
        self.xmean = np.dot(self.arx[self.arindex[:self.mu]].T, self.weights)
        self.zmean = np.dot(self.arz[self.arindex[:self.mu]].T, self.weights)
        self.ps = np.dot((1 - self.cs), self.ps) + np.dot((np.sqrt(self.cs * (2 - self.cs) * self.mueff)), np.dot(self.B, self.zmean))
        self.hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * self.counteval / self.lamda)) / self.chiN < 1.4 + 2 / (self.N + 1)
        self.pc = np.dot((1 - self.cc), self.pc) + np.dot(np.dot(self.hsig, np.sqrt(self.cc * (2 - self.cc) * self.mueff)), np.dot(np.dot(self.B, self.D), self.zmean))
        # adapt covariance matrix C
        self.C = np.dot((1 - self.c1 - self.cmu), self.C) \
            + np.dot(self.c1, ((self.pc * self.pc.T)
            + np.dot((1 - self.hsig) * self.cc * (2 - self.cc), self.C))) \
            + np.dot(self.cmu,
                     np.dot(np.dot(np.dot(np.dot(self.B, self.D), self.arz[self.arindex[:self.mu]].T),
                            np.diag(self.weights)), (np.dot(np.dot(self.B, self.D), self.arz[self.arindex[:self.mu]].T)).T))
        # adapt step size sigma
        self.sigma = self.sigma * np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
        # diagonalization
        if self.counteval - self.eigenval > self.lamda / (self.c1 + self.cmu) / self.N / 10:
            self.eigenval = self.counteval
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            self.D, self.B = np.linalg.eig(self.C)
            self.D = np.diag(np.sqrt(self.D))
        #history
        self.history['short_best'].append(arfitness[0])
        if len(self.history['short_best']) >= self.short_history_len:
            self.history['short_best'].popleft()
        if self.generation % 5 == 0:  # last 20 %
            self.history['long_best'].append(arfitness[0])
            self.history['long_median'].append(np.median(arfitness))
        if len(self.history['long_best']) >= self.long_history_len_up:
            self.history['long_best'].popleft()
            self.history['long_median'].popleft()
        self.checkStop()
        if self.generation % 20 == 0:
            self.logState()

    def fmin(self):
        """
        Method for actual function minimization. Iterates while not end.
        If unsuccess termination criteria is met then the method is restarted
        with doubled population. If the number of maximum evaluations is
        reached or the function is acceptable minimized, iterations ends and
        result is returned.

        Returns:
            scipy.optimize.Result : result of minimization.
        """
        while self.status != 0 and self.status != 1:
            if self.status > 2:
                logging.warning("Restart due to %s", self.stop_criteria[self.status] )
                self.restart(2)
            pop = self.newGeneration()
            values = np.empty(pop.shape[0])
            for i in xrange(pop.shape[0]):
                values[i] = self.func(pop[i])
            self.update(values)
        return self.result
    
    def restart(self, lamda_factor):
        """
        Restart whole method to initial state, but with population multiplied
        by lamda_factor.
        """
        self.initVariables(self.store_parameters['sigma'], np.random.rand(self.N), lamda_factor=lamda_factor)

    def checkStop(self):
        """
        Termination criteria of method. They are checked every iteration.
        """
        i = self.generation % self.N
        self.stop_conditions = (self.arfitness[0] <= self.stopfitness,
                                self.counteval > self.stopeval,
                                sum(self.xmean == self.xmean + 0.1 * self.sigma * self.D[i] * self.B[:, i]) == self.N,
                                np.any(self.xmean == self.xmean + 0.2 * self.sigma * np.sqrt(np.diag(self.C))),
                                #len(self.history['long_median']) > self.long_history_len_down and \
                                #np.median(list(itertools.islice(self.history['long_median'], int(0.7*len(self.history['long_median'])), None))) <= \
                                #np.median(list(itertools.islice(self.history['long_median'],int(0.3*len(self.history['long_median']))))),
                                np.linalg.cond(self.C) > self.condition_cov_max,
                                self.sigma * np.max(self.D) >= self.tolxup,
                                max(self.history['short_best']) - min(self.history['short_best']) <= self.tolfun and self.arfitness[-1] - self.arfitness[0] <= self.tolfun,
                                np.all(self.sigma * self.pc < self.tolx) and np.all(self.sigma * np.sqrt(np.diag(self.C)) < self.tolx)
                               )
        if np.any(self.stop_conditions):
            self.status = self.stop_conditions.index(True)
        return True

    def logState(self):
        """
        Function for logging the progress of method.
        """
        logging.debug("generation: {generation:<5}, v: {v_function:<6.2e}, sigma: {sigma:.2e}, best: {best}, xmean: {xmean}".format(
            generation=self.generation, best=map(lambda x: round(x, 8), self.arx[self.arindex[0]]), v_function=self.arfitness[0], sigma=self.sigma, xmean=self.xmean))

    @property
    def result(self):
        """
        Result of method. Not returned while minimization is in progress.
        """
        if self.status < 0:
                raise AttributeError("Result is not ready yet, cmaes is not finished")
        else:
            self._result = scipy.optimize.Result()        
            self._result['x'] = self.arx[self.arindex[0]]
            self._result['fun'] = self.arfitness[0]
            self._result['nfev'] = self.counteval
            if self.status == 0:
                self._result['success'] = True
                self._result['status'] = self.status
                self._result['message'] = "Optimization terminated successfully."
            else:
                self._result['success'] = False
                self._result['status'] = self.status
                self._result['message'] = self.stop_criteria[self.status]
        return self._result


def fmin(func, N):
    """
    Function for easy call function minimization from other modules.
    """
    c = CMAES(func, N)
    return c.fmin()

