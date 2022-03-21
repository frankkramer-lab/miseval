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
#          Unittest: Edge Cases & Exceptions          #
#-----------------------------------------------------#
class TEST_EdgeCases(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Create identical ground truth & prediction
        np.random.seed(1)
        self.gt_identical = np.random.randint(2, size=(32,32))
        self.pd_identical = np.random.randint(2, size=(32,32))
        # Create normal ground truth & prediction
        np.random.seed(8)
        self.gt_normal = np.random.randint(2, size=(32,32))
        np.random.seed(16)
        self.pd_normal = np.random.randint(2, size=(32,32))
        # Create empty ground truth & prediction
        self.gt_empty = np.zeros((32,32))
        self.pd_empty = np.zeros((32,32))
        # Create full ground truth & prediction
        self.gt_full = np.full((32,32), 1)
        self.pd_full = np.full((32,32), 1)

    #-------------------------------------------------#
    #              EdgeCase : Variant #1              #
    #-------------------------------------------------#
    def test_EdgeCase_identical(self):
        for metric in metric_dict:
            scores = evaluate(self.gt_identical, self.pd_identical, metric,
                             multi_class=True, n_classes=2,
                             provided_prob=False)
            self.assertTrue(isinstance(scores[0], np.int64) or \
                            isinstance(scores[0], np.float64))
            self.assertTrue(isinstance(scores[1], np.int64) or \
                            isinstance(scores[1], np.float64))

    #-------------------------------------------------#
    #              EdgeCase : Variant #2              #
    #-------------------------------------------------#
    def test_EdgeCase_normal_vs_all(self):
        for metric in metric_dict:
            gt = self.gt_normal
            for pd in [self.pd_normal, self.pd_empty, self.pd_full]:
                scores = evaluate(gt, pd, metric,
                                  multi_class=True, n_classes=2,
                                  provided_prob=False)
                self.assertTrue(isinstance(scores[0], np.int64) or \
                                isinstance(scores[0], np.float64))
                self.assertTrue(isinstance(scores[1], np.int64) or \
                                isinstance(scores[1], np.float64))

    #-------------------------------------------------#
    #              EdgeCase : Variant #3              #
    #-------------------------------------------------#
    def test_EdgeCase_empty_vs_all(self):
        for metric in metric_dict:
            gt = self.gt_empty
            for pd in [self.pd_normal, self.pd_empty, self.pd_full]:
                scores = evaluate(gt, pd, metric,
                                  multi_class=True, n_classes=2,
                                  provided_prob=False)
                self.assertTrue(isinstance(scores[0], np.int64) or \
                                isinstance(scores[0], np.float64))
                self.assertTrue(isinstance(scores[1], np.int64) or \
                                isinstance(scores[1], np.float64))

    #-------------------------------------------------#
    #              EdgeCase : Variant #4              #
    #-------------------------------------------------#
    def test_EdgeCase_full_vs_all(self):
        for metric in metric_dict:
            gt = self.gt_full
            for pd in [self.pd_normal, self.pd_empty, self.pd_full]:
                scores = evaluate(gt, pd, metric,
                                  multi_class=True, n_classes=2,
                                  provided_prob=False)
                self.assertTrue(isinstance(scores[0], np.int64) or \
                                isinstance(scores[0], np.float64))
                self.assertTrue(isinstance(scores[1], np.int64) or \
                                isinstance(scores[1], np.float64))
