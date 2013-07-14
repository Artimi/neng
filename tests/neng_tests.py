from nose.tools import *
import neng
import unittest

#def setup():
    #print "SETUP!"

#def teardown():
    #print "TEAR DOWN!"

#def test_basic():
    #print "I RAN!"

class Test_strategy_profile(unittest.TestCase):

    def setUp(self):
        self.flat_profile = [2, 8, 2, 3, 5, 3, 3, 4]
        self.shape = [2, 3, 3]
        self.profile = neng.StrategyProfile(self.flat_profile, self.shape)

    def test_get(self):
        self.assertEqual(self.profile[0][0], self.flat_profile[0])
        self.assertEqual(self.profile[1][1], self.flat_profile[3])
        self.assertEqual(self.profile[2][2], self.flat_profile[7])
        self.assertEqual(self.profile[0], self.flat_profile[0:2])
        self.assertEqual(self.profile[1], self.flat_profile[2:5])
        self.assertEqual(self.profile[2], self.flat_profile[5:8])

    @raises(IndexError)
    def test_get_key_fail(self):
        self.profile[3][0]

    def test_set(self):
        self.profile[0] = [3,7]
        self.assertEqual(self.profile[0], [3,7])
        self.profile[1][0] = 3
        self.assertEqual(self.profile[1][0], 3)

    @raises(IndexError)
    def test_set_fail(self):
        # player 1 has only 2 strategies/options
        self.profile[0] = [3,3,3]

    def test_normalize(self):
        a = self.profile.normalize()
        self.assertEqual([[0.2, 0.8],[0.2, 0.3, 0.5],[0.3, 0.3, 0.4]], a)

    def test_str(self):
        self.assertEqual(str(self.profile), '2, 8, 2, 3, 5, 3, 3, 4')

