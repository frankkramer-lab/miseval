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
import numpy as np

#-----------------------------------------------------#
#            Calculate : Confusion Matrix             #
#-----------------------------------------------------#
def calc_ConfusionMatrix(truth, pred, c=1, dtype=np.int64, **kwargs):
    # Obtain predicted and actual condition
    gt = np.equal(truth, c)
    pd = np.equal(pred, c)
    not_gt = np.logical_not(gt)
    not_pd = np.logical_not(pd)
    # Compute Confusion Matrix
    tp = np.logical_and(pd, gt).sum()
    tn = np.logical_and(not_pd, not_gt).sum()
    fp = np.logical_and(pd, not_gt).sum()
    fn = np.logical_and(not_pd, gt).sum()
    # Convert to desired numpy type to avoid overflow
    tp = tp.astype(dtype)
    tn = tn.astype(dtype)
    fp = fp.astype(dtype)
    fn = fn.astype(dtype)
    # Return Confusion Matrix
    return tp, tn, fp, fn

#-----------------------------------------------------#
#              Calculate : True Positive              #
#-----------------------------------------------------#
def calc_TruePositive(truth, pred, c=1, **kwargs):
    # Obtain predicted and actual condition
    gt = np.equal(truth, c)
    pd = np.equal(pred, c)
    not_gt = np.logical_not(gt)
    not_pd = np.logical_not(pd)
    # Compute true positive
    tp = np.logical_and(pd, gt).sum()
    # Return true positive
    return tp

#-----------------------------------------------------#
#              Calculate : True Negative              #
#-----------------------------------------------------#
def calc_TrueNegative(truth, pred, c=1, **kwargs):
    # Obtain predicted and actual condition
    gt = np.equal(truth, c)
    pd = np.equal(pred, c)
    not_gt = np.logical_not(gt)
    not_pd = np.logical_not(pd)
    # Compute true negative
    tn = np.logical_and(not_pd, not_gt).sum()
    # Return true negative
    return tn

#-----------------------------------------------------#
#              Calculate : False Positive             #
#-----------------------------------------------------#
def calc_FalsePositive(truth, pred, c=1, **kwargs):
    # Obtain predicted and actual condition
    gt = np.equal(truth, c)
    pd = np.equal(pred, c)
    not_gt = np.logical_not(gt)
    not_pd = np.logical_not(pd)
    # Compute false positive
    fp = np.logical_and(pd, not_gt).sum()
    # Return false positive
    return fp

#-----------------------------------------------------#
#              Calculate : False Negative             #
#-----------------------------------------------------#
def calc_FalseNegative(truth, pred, c=1, **kwargs):
    # Obtain predicted and actual condition
    gt = np.equal(truth, c)
    pd = np.equal(pred, c)
    not_gt = np.logical_not(gt)
    not_pd = np.logical_not(pd)
    # Compute false negative
    fn = np.logical_and(not_pd, gt).sum()
    # Return false negative
    return fn
