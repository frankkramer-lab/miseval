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
#              Calculate : Cohen's Kappa              #
#-----------------------------------------------------#
"""
References:
    Cohen J. A Coefficient of Agreement for Nominal Scales.
    Educ Psychol Meas [Internet]. 1960 Apr 2 [cited 2022 Jan 8];
    20(1):37–46. Available from:
    http://journals.sagepub.com/doi/10.1177/001316446002000104
"""
def calc_Kappa(truth, pred, c=1, dtype=np.float64, **kwargs):
    # Obtain confusion mat
    tp, tn, fp, fn = calc_ConfusionMatrix(truth, pred, c, dtype)
    # Compute kappa
    fa = tp + tn
    fc = ((tn+fn)*(tn+fp) + (fp+tp)*(fn+tp)) / (tp+tn+fp+fn)
    if ((tp+tn+fp+fn)-fc) != 0 : kappa = (fa-fc) / ((tp+tn+fp+fn)-fc)
    else : kappa = 0.0
    # Return kappa score
    return kappa
