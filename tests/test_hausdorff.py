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
#             Unittest: Hausdorff Distance            #
#-----------------------------------------------------#
class TEST_HausdorffDistance(unittest.TestCase):
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
        #same for 3d
        # Create ground truth
        np.random.seed(1)
        self.gt_bi_3d = np.random.randint(2, size=(32,32,32))
        self.gt_mc_3d = np.random.randint(5, size=(32,32,32))
        # Create prediction
        np.random.seed(2)
        self.pd_bi_3d = np.random.randint(2, size=(32,32,32))
        self.pd_mc_3d = np.random.randint(5, size=(32,32,32))

    #-------------------------------------------------#
    #      Calculate : Simple Hausdorff Distance      #
    #-------------------------------------------------#
    def test_calc_SimpleHausdorffDistance(self):
        # Check binary score
        score_bi = calc_SimpleHausdorffDistance(self.gt_bi, self.pd_bi, c=1)
        self.assertTrue(isinstance(score_bi, np.float64))
        # Check multi-class score
        for i in range(5):
            score_mc = calc_SimpleHausdorffDistance(self.gt_mc, self.pd_mc, c=i)
            self.assertTrue(isinstance(score_mc, np.float64))
        # Check existance in metric_dict
        self.assertTrue("HD" in metric_dict)
        self.assertTrue(callable(metric_dict["HD"]))
        self.assertTrue("HausdorffDistance" in metric_dict)
        self.assertTrue(callable(metric_dict["HausdorffDistance"]))

    #-------------------------------------------------#
    #     Calculate : Average Hausdorff Distance      #
    #-------------------------------------------------#
    def test_calc_AverageHausdorffDistance(self):
        # Check binary score
        score_bi = calc_AverageHausdorffDistance(self.gt_bi, self.pd_bi, c=1)
        self.assertTrue(isinstance(score_bi, np.float64))
        # Check multi-class score
        for i in range(5):
            score_mc = calc_AverageHausdorffDistance(self.gt_mc, self.pd_mc, c=i)
            self.assertTrue(isinstance(score_mc, np.float64))
        # Check existance in metric_dict
        self.assertTrue("AHD" in metric_dict)
        self.assertTrue(callable(metric_dict["AHD"]))
        self.assertTrue("AverageHausdorffDistance" in metric_dict)
        self.assertTrue(callable(metric_dict["AverageHausdorffDistance"]))

    #-------------------------------------------------#
    #     Calculate : Average Hausdorff Distance 3d   #
    #-------------------------------------------------#
    def test_calc_AverageHausdorffDistance_3d(self):
        # Check binary score
        score_bi = calc_AverageHausdorffDistance(self.gt_bi_3d, self.pd_bi_3d, c=1)
        self.assertTrue(isinstance(score_bi, np.float64))
        # Check multi-class score
        for i in range(5):
            score_mc = calc_AverageHausdorffDistance(self.gt_mc_3d, self.pd_mc_3d, c=i)
            self.assertTrue(isinstance(score_mc, np.float64))
