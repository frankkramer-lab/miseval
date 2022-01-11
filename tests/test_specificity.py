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
#                Unittest: Specificity                #
#-----------------------------------------------------#
class TEST_Specificity(unittest.TestCase):
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
    #         Calculate : Specificity via Sets        #
    #-------------------------------------------------#
    def test_calc_Specificity_Sets(self):
        # Check binary score
        score_bi = calc_Specificity_Sets(self.gt_bi, self.pd_bi, c=1)
        self.assertTrue(isinstance(score_bi, np.float64))
        # Check multi-class score
        for i in range(5):
            score_mc = calc_Specificity_Sets(self.gt_mc, self.pd_mc, c=i)
            self.assertTrue(isinstance(score_mc, np.float64))
        # Check existance in metric_dict
        self.assertTrue("SPEC" in metric_dict)
        self.assertTrue(callable(metric_dict["SPEC"]))
        self.assertTrue("Specificity" in metric_dict)
        self.assertTrue(callable(metric_dict["Specificity"]))
        self.assertTrue("TNR" in metric_dict)
        self.assertTrue(callable(metric_dict["TNR"]))
        self.assertTrue("TrueNegativeRate" in metric_dict)
        self.assertTrue(callable(metric_dict["TrueNegativeRate"]))

    #-------------------------------------------------#
    #          Calculate : Specificity via CM         #
    #-------------------------------------------------#
    def test_calc_Specificity_CM(self):
        # Check binary score
        score_bi = calc_Specificity_CM(self.gt_bi, self.pd_bi, c=1)
        self.assertTrue(isinstance(score_bi, np.float64))
        self.assertTrue(score_bi == calc_Specificity(self.gt_bi, self.pd_bi, c=1))
        # Check multi-class score
        for i in range(5):
            score_mc = calc_Specificity_CM(self.gt_mc, self.pd_mc, c=i)
            self.assertTrue(isinstance(score_mc, np.float64))
            self.assertTrue(score_mc == calc_Specificity(self.gt_mc, self.pd_mc, c=i))
