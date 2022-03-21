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
#          Calculate : Volumetric Similarity          #
#-----------------------------------------------------#
"""
Formula:
    VS = 1 - (|FN-FP| / (2TP + FP + FN))

References:
    Taha, A.A., Hanbury, A.
    Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool.
    BMC Med Imaging 15, 29 (2015). https://doi.org/10.1186/s12880-015-0068-x
"""
def calc_VolumetricSimilarity(truth, pred, c=1, **kwargs):
    # Obtain confusion mat
    tp, tn, fp, fn = calc_ConfusionMatrix(truth, pred, c)
    # Compute VS
    if (2*tp + fp + fn) != 0:
        vs = 1 - (np.abs(fn-fp) / (2*tp + fp + fn))
    else : vs = 1.0 - 0.0
    # Return VS score
    return vs
