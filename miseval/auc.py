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
from sklearn.metrics import roc_auc_score
# Internal modules
from miseval.confusion_matrix import calc_ConfusionMatrix

#-----------------------------------------------------#
#            Calculate : AUC via trapezoid            #
#-----------------------------------------------------#
"""
Formula:
    AUC = 1 - 1/2 * (FP/(FP+TN) + FN/(FN+TP))

References:
    Powers DMW. Evaluation: from precision, recall and F-measure to ROC, informedness, markedness and correlation.
    2020 Oct 10 [cited 2022 Jan 8]; Available from: http://arxiv.org/abs/2010.16061
"""
def calc_AUC_trapezoid(truth, pred, c=1, **kwargs):
    # Obtain confusion mat
    tp, tn, fp, fn = calc_ConfusionMatrix(truth, pred, c)
    # Compute AUC
    if (fp+tn) != 0 : x = fp/(fp+tn)
    else : x = 0.0
    if (fn+tp) != 0 : y = fn/(fn+tp)
    else : y = 0.0
    auc = 1 - (1/2)*(x + y)
    # Return AUC
    return auc

#-----------------------------------------------------#
#           Calculate : AUC via probability           #
#-----------------------------------------------------#
def calc_AUC_probability(truth, pred_prob, c=1, rounding_precision=5, **kwargs):
    # Round probability to reduce unnecessary thresholds
    prob = np.round(pred_prob[:,:,c], rounding_precision)
    # Obtain ground truth set with associated class
    gt = np.equal(truth, c).astype(int)
    auc = roc_auc_score(gt.flatten(), prob.flatten())
    # Return AUC
    return auc
