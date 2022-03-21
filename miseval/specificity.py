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
# Internal modules
from miseval.confusion_matrix import calc_ConfusionMatrix

#-----------------------------------------------------#
#           Calculate : Specificity via Sets          #
#-----------------------------------------------------#
def calc_Specificity_Sets(truth, pred, c=1, **kwargs):
    # Obtain sets with associated class
    not_gt = np.logical_not(np.equal(truth, c))
    not_pd = np.logical_not(np.equal(pred, c))
    # Calculate specificity
    if (not_gt).sum() != 0:
        spec = np.logical_and(not_pd, not_gt).sum() / (not_gt).sum()
    else : spec = 0.0
    # Return specificity
    return spec

#-----------------------------------------------------#
#            Calculate : Specificity via CM           #
#-----------------------------------------------------#
def calc_Specificity_CM(truth, pred, c=1, **kwargs):
    # Obtain confusion matrix
    tp, tn, fp, fn = calc_ConfusionMatrix(truth, pred, c)
    # Calculate specificity
    if (tn + fp) != 0 : spec = (tn) / (tn + fp)
    else : spec = 0.0
    # Return specificity
    return spec
