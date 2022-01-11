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
from sklearn.metrics import balanced_accuracy_score, adjusted_rand_score
# Internal modules
from miseval.confusion_matrix import calc_ConfusionMatrix

#-----------------------------------------------------#
#            Calculate : Accuracy via Sets            #
#-----------------------------------------------------#
def calc_Accuracy_Sets(truth, pred, c=1):
    try:
        # Obtain sets with associated class
        gt = np.equal(truth, c)
        pd = np.equal(pred, c)
        not_gt = np.logical_not(gt)
        not_pd = np.logical_not(pd)
        # Calculate Accuracy
        acc = (np.logical_and(pd, gt).sum() + \
               np.logical_and(not_pd, not_gt).sum()) / gt.size
    except ZeroDivisionError : acc = 0.0
    # Return computed Accuracy
    return acc

#-----------------------------------------------------#
#           Calculate : Accuracy via ConfMat          #
#-----------------------------------------------------#
def calc_Accuracy_CM(truth, pred, c=1):
    try:
        # Obtain confusion mat
        tp, tn, fp, fn = calc_ConfusionMatrix(truth, pred, c)
        # Calculate Accuracy
        acc = (tp + tn) / (tp + tn + fp + fn)
    except ZeroDivisionError : acc = 0.0
    # Return computed Accuracy
    return acc

#-----------------------------------------------------#
#            Calculate : Balanced Accuracy            #
#-----------------------------------------------------#
""" BACC = (Sensitivity + Specificity) / 2          """
def calc_BalancedAccuracy(truth, pred, c=1):
    # Obtain sets with associated class
    gt = np.equal(truth, c).flatten()
    pd = np.equal(pred, c).flatten()
    # Compute BACC via scikit-learn
    return balanced_accuracy_score(gt, pd)

#-----------------------------------------------------#
#           Calculate : Adjusted Rand Index           #
#-----------------------------------------------------#
""" ARI = (RI - Expected_RI) / (max(RI) - Expected_RI) """
def calc_AdjustedRandIndex(truth, pred, c=1):
    # Obtain sets with associated class
    gt = np.equal(truth, c).flatten()
    pd = np.equal(pred, c).flatten()
    # Compute ARI via scikit-learn
    return adjusted_rand_score(gt, pd)
