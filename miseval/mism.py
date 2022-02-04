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
import math
# Internal modules
from miseval.confusion_matrix import calc_ConfusionMatrix
from miseval import calc_DSC_Sets, calc_Specificity_Weighted

#-----------------------------------------------------#
#                Calculate : MISmetric                #
#-----------------------------------------------------#
"""
Combination of weighted Specificity for p=0 and Dice Similarity Coefficient
as Backbone for p>0.

Recommended for weak-labeled datasets.

References:
    Coming soon.
"""
def calc_MISm(truth, pred, c=1, alpha=0.1):
    # Obtain confusion mat
    tp, tn, fp, fn = calc_ConfusionMatrix(truth, pred, c)
    # Identify metric wing
    p = tp + fn
    # Compute & return normal dice if p > 0
    if p > 0: return calc_DSC_Sets(truth, pred, c)
    # Compute & return weighted specificity if p = 0
    else : return calc_Specificity_Weighted(truth, pred, c, alpha)
