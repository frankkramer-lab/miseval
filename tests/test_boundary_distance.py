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
#             Unittest: Boundary Distance             #
#-----------------------------------------------------#
class TEST_BoundaryDistance(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Create ground truth
        np.random.seed(1)
        self.gt_bi = np.random.randint(2, size=(16,16))
        self.gt_mc = np.random.randint(5, size=(16,16))
        # Create prediction mask
        np.random.seed(2)
        self.pd_bi = np.random.randint(2, size=(16,16))
        self.pd_mc = np.random.randint(5, size=(16,16))
        # Create prediction probability
        self.prob_bi = np.random.rand(16,16,2)
        self.prob_mc = np.random.rand(16,16,5)

    #-------------------------------------------------#
    #          Calculate : Boundary Distance          #
    #-------------------------------------------------#
    def test_calc_BoundaryDistance_General(self):
        # Check binary score
        score_bi = calc_Boundary_Distance(self.gt_bi, self.pd_bi, c=1)
        self.assertTrue(isinstance(score_bi, np.float64))
        # Check multi-class score
        for i in range(5):
            score_mc = calc_Boundary_Distance(self.gt_mc, self.pd_mc, c=i)
            self.assertTrue(isinstance(score_mc, np.float64))
        # Check existance in metric_dict
        self.assertTrue("BoundaryDistance" in metric_dict)
        self.assertTrue(callable(metric_dict["BoundaryDistance"]))
        self.assertTrue("Distance" in metric_dict)
        self.assertTrue(callable(metric_dict["Distance"]))
        self.assertTrue("BD" in metric_dict)
        self.assertTrue(callable(metric_dict["BD"]))

    #-------------------------------------------------#
    #      Boundary Distance : Distance Function      #
    #-------------------------------------------------#
    def test_calc_BoundaryDistance_DistanceFunction(self):
        # Define list of pooling function
        df_list = ["bhattacharyya", "bhattacharyya_coefficient", "canberra",
                   "chebyshev", "chi_square", "cosine", "euclidean", "hamming",
                   "jensen_shannon", "kullback_leibler", "mae", "manhattan",
                   "minkowsky", "mse", "pearson", "squared_variation"]
        # Test boundary distance with each pooling function
        for df in df_list:
            score_bi = calc_Boundary_Distance(self.gt_bi, self.pd_bi, c=1,
                                              distance=df)
            self.assertTrue(isinstance(score_bi, np.float64))

    #-------------------------------------------------#
    #       Boundary Distance : Pooling Function      #
    #-------------------------------------------------#
    def test_calc_BoundaryDistance_PoolingFunction(self):
        # Define list of pooling function
        pf_list = ["sum", "mean", "amin", "amax"]
        # Test boundary distance with each pooling function
        for pf in pf_list:
            score_bi = calc_Boundary_Distance(self.gt_bi, self.pd_bi, c=1,
                                              pooling=pf)
            self.assertTrue(isinstance(score_bi, np.float64))
