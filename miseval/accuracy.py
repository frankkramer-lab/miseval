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
import warnings
# Internal modules
from miseval.confusion_matrix import calc_ConfusionMatrix

#-----------------------------------------------------#
#            Calculate : Accuracy via Sets            #
#-----------------------------------------------------#
def calc_Accuracy_Sets(truth, pred, c=1, **kwargs):
    # Obtain sets with associated class
    gt = np.equal(truth, c)
    pd = np.equal(pred, c)
    not_gt = np.logical_not(gt)
    not_pd = np.logical_not(pd)
    # Calculate Accuracy
    acc = (np.logical_and(pd, gt).sum() + \
           np.logical_and(not_pd, not_gt).sum()) / gt.size
    # Return computed Accuracy
    return acc

#-----------------------------------------------------#
#           Calculate : Accuracy via ConfMat          #
#-----------------------------------------------------#
def calc_Accuracy_CM(truth, pred, c=1, **kwargs):
    # Obtain confusion mat
    tp, tn, fp, fn = calc_ConfusionMatrix(truth, pred, c)
    # Calculate Accuracy
    acc = (tp + tn) / (tp + tn + fp + fn)
    # Return computed Accuracy
    return acc

#-----------------------------------------------------#
#            Calculate : Balanced Accuracy            #
#-----------------------------------------------------#
"""
Formula:
    BACC = (Sensitivity + Specificity) / 2

References:
[1] Brodersen, K.H.; Ong, C.S.; Stephan, K.E.; Buhmann, J.M. (2010).
    The balanced accuracy and its posterior distribution.
    Proceedings of the 20th International Conference on Pattern Recognition, 3121-24.

[2] John. D. Kelleher, Brian Mac Namee, Aoife D’Arcy, (2015).
    Fundamentals of Machine Learning for Predictive Data Analytics: Algorithms, Worked Examples, and Case Studies.
    https://mitpress.mit.edu/books/fundamentals-machine-learning-predictive-data-analytics
"""
def calc_BalancedAccuracy(truth, pred, c=1, **kwargs):
    # Obtain sets with associated class
    gt = np.equal(truth, c).flatten()
    pd = np.equal(pred, c).flatten()
    # Compute BACC via scikit-learn
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bacc = balanced_accuracy_score(gt, pd)
    # Return BACC score
    return np.float64(bacc)

#-----------------------------------------------------#
#           Calculate : Adjusted Rand Index           #
#-----------------------------------------------------#
"""
Formula:
    ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)

References:
[1] L. Hubert and P. Arabie, Comparing Partitions, Journal of Classification 1985
    https://link.springer.com/article/10.1007%2FBF01908075


[2] D. Steinley, Properties of the Hubert-Arabie adjusted Rand index,
    Psychological Methods 2004

[3] https://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index
"""
def calc_AdjustedRandIndex(truth, pred, c=1, **kwargs):
    # Obtain sets with associated class
    gt = np.equal(truth, c).flatten()
    pd = np.equal(pred, c).flatten()
    # Compute ARI via scikit-learn
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ari = adjusted_rand_score(gt, pd)
    # Return ARI score
    return np.float64(ari)
