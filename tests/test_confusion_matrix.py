#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2022 IT-Infrastructure for Translational Medical Research,    #
#                University of Augsburg                                        #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#==============================================================================#
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External modules
import numpy as np
import unittest
# Internal modules
from miseval import *

#-----------------------------------------------------#
#              Unittest: Confusion Matrix             #
#-----------------------------------------------------#
class TEST_ConfusionMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Create ground truth
        np.random.seed(1)
        self.gt_bi = np.random.randint(2, size=(32,32))
        self.gt_mc = np.random.randint(5, size=(32,32))
        # Create prediction
        np.random.seed(2)
        self.pd_bi = np.random.randint(2, size=(32,32))
        self.pd_mc = np.random.randint(5, size=(32,32))

    #-------------------------------------------------#
    #          Calculate : Confusion Matrix           #
    #-------------------------------------------------#
    def test_calc_ConfusionMatrix(self):
        # Check binary score
        scores_bi = calc_ConfusionMatrix(self.gt_bi, self.pd_bi, c=1)
        self.assertTrue(isinstance(scores_bi[0], np.int64))
        self.assertTrue(isinstance(scores_bi[1], np.int64))
        self.assertTrue(isinstance(scores_bi[2], np.int64))
        self.assertTrue(isinstance(scores_bi[3], np.int64))
        # Check multi-class score
        for i in range(5):
            scores_mc = calc_ConfusionMatrix(self.gt_mc, self.pd_mc, c=i)
            self.assertTrue(isinstance(scores_mc[0], np.int64))
            self.assertTrue(isinstance(scores_mc[1], np.int64))
            self.assertTrue(isinstance(scores_mc[2], np.int64))
            self.assertTrue(isinstance(scores_mc[3], np.int64))

    #-------------------------------------------------#
    #            Calculate : True Positive            #
    #-------------------------------------------------#
    def test_calc_TruePositive(self):
        # Check binary score
        score_bi = calc_TruePositive(self.gt_bi, self.pd_bi, c=1)
        self.assertTrue(isinstance(score_bi, np.int64))
        # Check multi-class score
        for i in range(5):
            score_mc = calc_TruePositive(self.gt_mc, self.pd_mc, c=i)
            self.assertTrue(isinstance(score_mc, np.int64))
        # Check existance in metric_dict
        self.assertTrue("TruePositive" in metric_dict)
        self.assertTrue(callable(metric_dict["TruePositive"]))

    #-------------------------------------------------#
    #            Calculate : True Negative            #
    #-------------------------------------------------#
    def test_calc_TrueNegative(self):
        # Check binary score
        score_bi = calc_TrueNegative(self.gt_bi, self.pd_bi, c=1)
        self.assertTrue(isinstance(score_bi, np.int64))
        # Check multi-class score
        for i in range(5):
            score_mc = calc_TrueNegative(self.gt_mc, self.pd_mc, c=i)
            self.assertTrue(isinstance(score_mc, np.int64))
        # Check existance in metric_dict
        self.assertTrue("TrueNegative" in metric_dict)
        self.assertTrue(callable(metric_dict["TrueNegative"]))

    #-------------------------------------------------#
    #           Calculate : False Positive            #
    #-------------------------------------------------#
    def test_calc_FalsePositive(self):
        # Check binary score
        score_bi = calc_FalsePositive(self.gt_bi, self.pd_bi, c=1)
        self.assertTrue(isinstance(score_bi, np.int64))
        # Check multi-class score
        for i in range(5):
            score_mc = calc_FalsePositive(self.gt_mc, self.pd_mc, c=i)
            self.assertTrue(isinstance(score_mc, np.int64))
        # Check existance in metric_dict
        self.assertTrue("FalsePositive" in metric_dict)
        self.assertTrue(callable(metric_dict["FalsePositive"]))

    #-------------------------------------------------#
    #           Calculate : False Negative            #
    #-------------------------------------------------#
    def test_calc_FalseNegative(self):
        # Check binary score
        score_bi = calc_FalseNegative(self.gt_bi, self.pd_bi, c=1)
        self.assertTrue(isinstance(score_bi, np.int64))
        # Check multi-class score
        for i in range(5):
            score_mc = calc_FalseNegative(self.gt_mc, self.pd_mc, c=i)
            self.assertTrue(isinstance(score_mc, np.int64))
        # Check existance in metric_dict
        self.assertTrue("FalseNegative" in metric_dict)
        self.assertTrue(callable(metric_dict["FalseNegative"]))
