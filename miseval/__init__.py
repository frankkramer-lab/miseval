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
#                   Metric Imports                    #
#-----------------------------------------------------#
# Confusion Matrix
from miseval.confusion_matrix import calc_ConfusionMatrix, \
                                     calc_FalseNegative, calc_FalsePositive, \
                                     calc_TrueNegative, calc_TruePositive

#-----------------------------------------------------#
#         Access Functions to Metric Functions        #
#-----------------------------------------------------#
# Metric Dictionary
metric_dict = {
    "TruePositive": calc_TruePositive,
    "TrueNegative": calc_TrueNegative,
    "FalsePositive": calc_FalsePositive,
    "FalseNegative": calc_FalseNegative
}

#-----------------------------------------------------#
#                     Core Imports                    #
#-----------------------------------------------------#
from miseval.core import evaluate


# Note:
# some dict which says if metric needs argmax before passing if probabilities==True
