import unittest
import os.path
import itertools

import nose.tools
import numpy as np

import neng

THIS_FILE_PATH = os.path.dirname(__file__)
SELTEN_PATH = os.path.join(THIS_FILE_PATH, '../games/gambit.nfg')
TWO_PATH = os.path.join(THIS_FILE_PATH, '../games/2x2x2.nfg')
SEVEN_PATH = os.path.join(THIS_FILE_PATH, '../games/7.nfg')
PRISONERS_PATH = os.path.join(THIS_FILE_PATH, '../games/prisoners.nfg')

with open(SELTEN_PATH) as f:
    SELTEN_STR = f.read()
with open(TWO_PATH) as f:
    TWO_STR = f.read()
with open(SEVEN_PATH) as f:
    SEVEN_STR = f.read()
with open(PRISONERS_PATH) as f:
    PRISONERS_STR = f.read()


class NengTestCase(unittest.TestCase):
    def assertListofStrategyProfileAlmostEqual(self, list1, list2, tol=1e-4):
        self.assertEqual(len(list1), len(list2), "Lists are not equally long.")
        for sp1, sp2 in zip(list1, list2):
            self.assertStrategyProfileAlmostEqual(sp1, sp2, tol)

    def assertStrategyProfileAlmostEqual(self, sp1, sp2, tol=1e-4):
        self.assertListEqual(sp1.shape, sp2.shape, "Shape is not equal")
        for sp1_pl, sp2_pl in zip(sp1._list, sp2._list):
            for a, b in zip(sp1_pl, sp2_pl):
                self.assertAlmostEqual(a, b, delta=tol)

    def assertListOfArraysEqual(self, list1, list2):
        self.assertEqual(len(list1), len(list2), "Lists are not equally long.")
        for arr1, arr2 in zip(list1, list2):
            np.testing.assert_array_equal(arr1, arr2)

    def assertDictsWithArraysEqual(self, d1, d2):
        self.assertEqual(set(d1.keys()), set(d2.keys()))
        for key in d1.keys():
            if isinstance(d1[key], list) and isinstance(d1[key][0], np.ndarray):
                self.assertListOfArraysEqual(d1[key], d2[key])
            else:
                self.assertEqual(d1[key], d2[key])


class Test_strategy_profile(NengTestCase):
    def setUp(self):
        self.flat_profile = [0.2, 0.8, 0.2, 0.3, 0.5, 0.3, 0.3, 0.4]
        self.shape = [2, 3, 3]
        self.profile = neng.StrategyProfile(self.flat_profile, self.shape)

    def test_get(self):
        self.assertEqual(self.profile[0][0], self.flat_profile[0])
        self.assertEqual(self.profile[1][1], self.flat_profile[3])
        self.assertEqual(self.profile[2][2], self.flat_profile[7])
        np.testing.assert_array_equal(self.profile[0], np.array(self.flat_profile[0:2]))
        np.testing.assert_array_equal(self.profile[1], np.array(self.flat_profile[2:5]))
        np.testing.assert_array_equal(self.profile[2], np.array(self.flat_profile[5:8]))

    @nose.tools.raises(IndexError)
    def test_get_key_fail(self):
        self.profile[3][0]

    def test_set(self):
        self.profile[0] = np.array([0.3, 0.7])
        self.assertTrue((self.profile[0] == np.array([0.3, 0.7])).all())
        self.profile[1][0] = 0.3
        self.assertEqual(self.profile[1][0], 0.3)

    @nose.tools.raises(IndexError)
    def test_set_fail(self):
        # player 1 has only 2 strategies/options
        self.profile[0] = [3, 3, 3]

    def test_coordinate(self):
        coordinate = [0, 1, 2]
        coordinated_profile = neng.StrategyProfile(coordinate, self.shape, coordinate=True)
        flat_right_profile = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        right_profile = neng.StrategyProfile(flat_right_profile, self.shape)
        self.assertEqual(coordinated_profile, right_profile)

    def test_normalize(self):
        denormalized_flat = [0.2, 0.8, 0.2, 0.3, 0.5, 0.3, 0.3, 0.4]
        denormalized_profile = neng.StrategyProfile(denormalized_flat, self.shape)
        a = denormalized_profile.normalize()
        self.assertListOfArraysEqual([np.array([0.2, 0.8]), np.array([0.2, 0.3, 0.5]), np.array([0.3, 0.3, 0.4])],
                                     a._list)

    def test_str(self):
        self.assertEqual(str(self.profile), '0.2, 0.8, 0.2, 0.3, 0.5, 0.3, 0.3, 0.4')

    def test_copy(self):
        copy_profile = self.profile.copy()
        copy_profile.normalize()
        self.assertNotEqual(copy_profile, self.profile)

    def test_updateWithPureStrategy(self):
        updated_flat = self.flat_profile[:]
        updated_flat[0] = 1.0
        updated_flat[1] = 0.0
        updated_profile = neng.StrategyProfile(updated_flat, self.shape)
        self.profile.updateWithPureStrategy(0, 0)
        self.assertEqual(updated_profile, self.profile)


class Test_GameReader(NengTestCase):
    def setUp(self):
        self.selten_array = [np.array([[1., 1.],
                                       [0., 0.],
                                       [0., 2.]]),
                             np.array([[1., 1.],
                                       [2., 3.],
                                       [2., 0.]])]
        self.selten_game = {'name': 'Selten (IJGT, 75), Figure 2, normal form',
                            'players': ['Player 1', 'Player 2'],
                            'num_players': 2,
                            'shape': [3, 2],
                            'sum_shape': 5,
                            'array': self.selten_array}

        self.two_array = [np.array([[[9., 0.],
                                     [0., 3.]],
                                    [[0., 3.],
                                     [9., 0.]]]),
                          np.array([[[8., 0.],
                                     [0., 4.]],
                                    [[0., 4.],
                                     [8., 0.]]]),
                          np.array([[[12., 0.],
                                     [0., 6.]],
                                    [[0., 6.],
                                     [2., 0.]]])]
        self.two_game = {'name': '2x2x2 Example from McKelvey-McLennan, with 9 Nash equilibria, 2 totally mixed',
                         'players': ["Player 1", "Player 2", "Player 3"],
                         'num_players': 3,
                         'shape': [2, 2, 2],
                         'sum_shape': 6,
                         'array': self.two_array}

    def test_readPayoff(self):
        gr = neng.GameReader()
        result = gr.readStr(SELTEN_STR)
        self.assertDictsWithArraysEqual(self.selten_game, result)

    def test_readOutcome(self):
        gr = neng.GameReader()
        result = gr.readStr(TWO_STR)
        self.assertDictsWithArraysEqual(self.two_game, result)

    def test_readFile(self):
        grFile = neng.GameReader()
        resultFile = grFile.readFile(SELTEN_PATH)
        grStr = neng.GameReader()
        resultStr = grStr.readStr(SELTEN_STR)
        self.assertDictsWithArraysEqual(resultFile, resultStr)

    def test_read(self):
        resultFile = neng.game_reader.read(SELTEN_PATH)
        resultStr = neng.game_reader.read(SELTEN_STR)
        self.assertDictsWithArraysEqual(resultFile, resultStr)


class Test_Game(NengTestCase):
    def setUp(self):
        self.game_selten = neng.Game(SELTEN_STR)
        self.game_selten_penalization = neng.Game(SELTEN_STR, trim='penalization')
        self.game_selten_str = """NFG 1 R "Selten (IJGT, 75), Figure 2, normal form"
{ "Player 1" "Player 2" } { 3 2 }

1.0 1.0 0.0 2.0 0.0 2.0 1.0 1.0 0.0 3.0 2.0 0.0"""
        self.game_selten_shape = [3, 2]
        self.game_selten_pne = [neng.StrategyProfile([1.0, 0.0, 0.0, 1.0, 0.0], self.game_selten_shape)]
        self.game_selten_mne = [[1.0, 0.0, 0.0, 1.0, 0.0],
                                [1.0, 0.0, 0.0, 0.5, 0.5]]
        self.game_selten_mne_profile = map(lambda x: neng.StrategyProfile(x, self.game_selten_shape),
                                           self.game_selten_mne)
        self.game_selten_not_mne = [[0.3, 0.5, 0.2, 0.1, 0.9],
                                    [0.1, 0.9, 0.0, 0.4, 0.6],
                                    [0.9, 0.1, 0.0, 0.33, 0.66],
                                    [0.0, 1.0, 0.0, 1.0, 0.0],
                                    ]
        self.game_selten_brs = {0: {(0, 1): set([(2, 1)]),
                                    (0, 0): set([(0, 0)]),
                                    (2, 1): set([(2, 1)]),
                                    (2, 0): set([(0, 0)]),
                                    (1, 0): set([(0, 0)]),
                                    (1, 1): set([(2, 1)])},
                                1: {(0, 0): set([(0, 1), (0, 0)]),
                                    (0, 1): set([(0, 1), (0, 0)]),
                                    (1, 0): set([(1, 1)]),
                                    (1, 1): set([(1, 1)]),
                                    (2, 0): set([(2, 0)]),
                                    (2, 1): set([(2, 0)])}}

        self.game_two = neng.Game(TWO_STR)
        self.game_two_array = [np.array([[[9., 0.],
                                          [0., 3.]],
                                         [[0., 3.],
                                          [9., 0.]]]),
                               np.array([[[8., 0.],
                                          [0., 4.]],
                                         [[0., 4.],
                                          [8., 0.]]]),
                               np.array([[[12., 0.],
                                          [0., 6.]],
                                         [[0., 6.],
                                          [2., 0.]]])]
        self.game_two_str = """NFG 1 R "2x2x2 Example from McKelvey-McLennan, with 9 Nash equilibria, 2 totally mixed"
{ "Player 1" "Player 2" "Player 3" } { 2 2 2 }

9.0 8.0 12.0 0.0 0.0 0.0 0.0 0.0 0.0 9.0 8.0 2.0 0.0 0.0 0.0 3.0 4.0 6.0 3.0 4.0 6.0 0.0 0.0 0.0"""
        self.game_two_shape = [2, 2, 2]
        self.game_two_pne = [[0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                             [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                             [0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
                             [1.0, 0.0, 0.0, 1.0, 0.0, 1.0]]
        self.game_two_pne = map(lambda profile: neng.StrategyProfile(profile, self.game_two_shape),
                                self.game_two_pne)
        self.game_two_dominated_strategies = [[], [], []]

        self.game_seven = neng.Game(SEVEN_STR)
        self.game_seven_shape = [7, 7]
        self.game_seven_mne = [[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.13143631436314362, 0.0, 0.0, 0.0, 0.0, 0.86856368563685649,
                                0.5818639798488664,
                                0.41813602015113355, 0.0, 0.0, 0.0, 0.0, 0.0]]
        self.game_seven_mne = map(lambda x: neng.StrategyProfile(x, self.game_seven_shape), self.game_seven_mne)

        self.prisoners = neng.Game(PRISONERS_STR)
        self.prisoners_dominated_strategies = [[1], [1]]
        self.prisoners_IESDS_deleted_strategies = [np.array([1]), np.array([1])]
        self.prisoners_IESDS_array = [np.array([[-10.]]), np.array([[-10.]])]

    def test_bestResponse(self):
        for player in range(self.game_selten.num_players):
            for coordinate in itertools.product(range(self.game_selten_shape[0]), range(self.game_selten_shape[1])):
                self.assertEqual(self.game_selten_brs[player][coordinate],
                                 self.game_selten.bestResponse(player, coordinate))

    def test_pne(self):
        self.assertEqual(self.game_selten.findEquilibria('pne'), self.game_selten_pne)
        self.assertEqual(self.game_two.findEquilibria('pne'), self.game_two_pne)

    def test_getDominatedStrategies(self):
        self.assertEqual(self.prisoners.getDominatedStrategies(), self.prisoners_dominated_strategies)
        self.assertEqual(self.game_two.getDominatedStrategies(), self.game_two_dominated_strategies)

    def test_IESDS(self):
        self.prisoners.IESDS()
        self.assertListOfArraysEqual(self.prisoners_IESDS_array, self.prisoners.array)
        self.assertEqual(self.prisoners_IESDS_deleted_strategies, self.prisoners.deleted_strategies)
        self.game_two.IESDS()
        self.assertListOfArraysEqual(self.game_two.array, self.game_two_array)

    def test_isDegenerate(self):
        self.assertTrue(self.game_selten.isDegenerate())
        self.assertFalse(self.game_two.isDegenerate())
        self.assertFalse(self.game_seven.isDegenerate())
        self.assertFalse(self.prisoners.isDegenerate())

    def test_LyapunovFunction_normalization(self):
        # normalization - assert that lf(ne) == 0.0, assert that lf(multiple of ne) == 0.0
        # assert that not ne > 0.0
        for ne in self.game_selten_mne:
            for i in [-1.5, -1, -0.5, 0.5, 0.33, 0.75, 1.0, 2]:
                modified_ne = [item * i for item in ne]
                np.testing.assert_almost_equal(self.game_selten.LyapunovFunction(modified_ne), 0.0)
        for nne in self.game_selten_not_mne:
            self.assertGreater(self.game_selten.LyapunovFunction(nne), 0.0)

    def test_LyapunovFunction_penalization(self):
        # penalization - assert that lf(ne) == 0.0, assert that lf(multiple of ne) > 0.0
        # assert that not ne > 0.0
        for ne in self.game_selten_mne:
            np.testing.assert_almost_equal(self.game_selten_penalization.LyapunovFunction(ne), 0.0)
            for i in [-1.5, -1, -0.5, 0.5, 0.33, 0.75, 2]:
                modified_ne = [item * i for item in ne]
                self.assertGreater(self.game_selten_penalization.LyapunovFunction(modified_ne), 0.0)
        for nne in self.game_selten_not_mne:
            self.assertGreater(self.game_selten_penalization.LyapunovFunction(nne), 0.0)

    def test_str(self):
        self.assertEqual(self.game_selten_str, str(self.game_selten))
        self.assertEqual(self.game_two_str, str(self.game_two))

    def test_support_enumeration(self):
        self.assertListofStrategyProfileAlmostEqual(self.game_seven.findEquilibria('support_enumeration'),
                                                    self.game_seven_mne)
        self.assertListofStrategyProfileAlmostEqual(self.game_selten.findEquilibria('support_enumeration'),
                                                    self.game_selten_mne_profile)
