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
#           Unittest: Entropy-based metrics           #
#-----------------------------------------------------#
class TEST_Entropy(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Create ground truth
        np.random.seed(1)
        self.gt_bi = np.random.randint(2, size=(32,32))
        self.gt_mc = np.random.randint(5, size=(32,32))
        # Create prediction mask
        np.random.seed(2)
        self.pd_bi = np.random.randint(2, size=(32,32))
        self.pd_mc = np.random.randint(5, size=(32,32))
        # Create prediction probability
        self.prob_bi = np.random.rand(32,32,2)
        self.prob_mc = np.random.rand(32,32,5)

    #-------------------------------------------------#
    #            Calculate : Cross-Entropy            #
    #-------------------------------------------------#
    def test_calc_CrossEntropy(self):
        # Check binary score
        score_bi = calc_CrossEntropy(self.gt_bi, self.prob_bi, c=1,
                                     provided_prob=True)
        self.assertTrue(isinstance(score_bi, np.float64))
        # Check multi-class score
        for i in range(5):
            score_mc = calc_CrossEntropy(self.gt_mc, self.prob_mc, c=i,
                                         provided_prob=True)
            self.assertTrue(isinstance(score_mc, np.float64))
        # Check existance in metric_dict
        self.assertTrue("CE" in metric_dict)
        self.assertTrue(callable(metric_dict["CE"]))
        self.assertTrue("CrossEntropy" in metric_dict)
        self.assertTrue(callable(metric_dict["CrossEntropy"]))
