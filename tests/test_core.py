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
from miseval import evaluate, metric_dict

#-----------------------------------------------------#
#             Unittest: Core Evaluate Hub             #
#-----------------------------------------------------#
class TEST_CoreEvaluate(unittest.TestCase):
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
    #         Evaluate : Metric Functionality         #
    #-------------------------------------------------#
    def test_evaluate_metric(self):
        # Check miseval function callings
        for metric in metric_dict:
            self.assertTrue(metric in metric_dict)
            self.assertTrue(callable(metric_dict[metric]))
            evaluate(self.gt_bi, self.pd_bi, metric)
            evaluate(self.gt_bi, self.pd_bi, metric_dict[metric])
        # Check custom function
        def my_custom_function(g,p,c) : return 0.0
        evaluate(self.gt_bi, self.pd_bi, my_custom_function)
        # Check lower string metric calling
        evaluate(self.gt_bi, self.pd_bi, "tp")
        evaluate(self.gt_bi, self.pd_bi, "fn")
        # Provide some non-existent metric string
        self.assertRaises(KeyError, evaluate, self.gt_bi, self.pd_bi, "test")
        # Provide a non callable metric
        self.assertRaises(ValueError, evaluate, self.gt_bi, self.pd_bi, 7)

    #-------------------------------------------------#
    #          Evaluate : Binary with Binary          #
    #-------------------------------------------------#
    def test_evaluate_binary_bi(self):
        for metric in metric_dict:
            score = evaluate(self.gt_bi, self.pd_bi, metric,
                             multi_class=False, n_classes=2,
                             provided_prob=False)
            self.assertTrue(isinstance(score, np.int64) or \
                            isinstance(score, np.float64))

    #-------------------------------------------------#
    #        Evaluate : Binary with Multi-Class       #
    #-------------------------------------------------#
    def test_evaluate_binary_mc(self):
        for metric in metric_dict:
            scores = evaluate(self.gt_mc, self.pd_mc, metric,
                              multi_class=False, n_classes=5,
                              provided_prob=False)
            self.assertTrue(len(scores) == 5)
            self.assertTrue(isinstance(scores, np.ndarray))

    #-------------------------------------------------#
    #        Evaluate : Multi-Class with Binary       #
    #-------------------------------------------------#
    def test_evaluate_multiclass_bi(self):
        for metric in metric_dict:
            scores = evaluate(self.gt_bi, self.pd_bi, metric,
                              multi_class=True, n_classes=2,
                              provided_prob=False)
            self.assertTrue(len(scores) == 2)
            self.assertTrue(isinstance(scores, np.ndarray))

    #-------------------------------------------------#
    #     Evaluate : Multi-Class with Multi-Class     #
    #-------------------------------------------------#
    def test_evaluate_multiclass_mc(self):
        for metric in metric_dict:
            scores = evaluate(self.gt_mc, self.pd_mc, metric,
                              multi_class=True, n_classes=5,
                              provided_prob=False)
            self.assertTrue(len(scores) == 5)
            self.assertTrue(isinstance(scores, np.ndarray))

    #-------------------------------------------------#
    #           Evaluate : Exception Testing          #
    #-------------------------------------------------#
    def test_evaluate_exceptions(self):
        # Check for incorrect number of classes in ground truth
        with self.assertRaises(ValueError):
            scores = evaluate(self.gt_mc, self.pd_bi, "TruePositive",
                              multi_class=True, n_classes=2,
                              provided_prob=False)
        # Check for incorrect number of classes in prediction
        with self.assertRaises(ValueError):
            scores = evaluate(self.gt_bi, self.pd_mc, "TruePositive",
                              multi_class=True, n_classes=2)
