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
from miseval.confusion_matrix import *
# Dice Similarity Coefficient
from miseval.dice import *
from miseval.dice import calc_DSC_Sets as calc_DSC
# Intersection-over-Union
from miseval.jaccard import *
from miseval.jaccard import calc_IoU_Sets as calc_IoU
# Accuracy
from miseval.accuracy import *
from miseval.accuracy import calc_Accuracy_Sets as calc_Accuracy

#-----------------------------------------------------#
#         Access Functions to Metric Functions        #
#-----------------------------------------------------#
# Metric Dictionary
metric_dict = {
    "TruePositive": calc_TruePositive,
    "TrueNegative": calc_TrueNegative,
    "FalsePositive": calc_FalsePositive,
    "FalseNegative": calc_FalseNegative,
    "TP": calc_TruePositive,
    "TN": calc_TrueNegative,
    "FP": calc_FalsePositive,
    "FN": calc_FalseNegative,
    "DSC": calc_DSC,
    "Dice": calc_DSC,
    "DiceSimilarityCoefficient": calc_DSC,
    "IoU": calc_IoU,
    "Jaccard": calc_IoU,
    "IntersectionOverUnion": calc_IoU,
    "ACC": calc_Accuracy,
    "Accuracy": calc_Accuracy,
    "RI": calc_Accuracy,
    "RandIndex": calc_Accuracy,
    "IntersectionOverUnion": calc_IoU,
    "IntersectionOverUnion": calc_IoU,
    "BACC": calc_BalancedAccuracy,
    "BalancedAccuracy": calc_BalancedAccuracy,
    "ARI": calc_AdjustedRandIndex,
    "AdjustedRandIndex": calc_AdjustedRandIndex
}

#-----------------------------------------------------#
#                     Core Imports                    #
#-----------------------------------------------------#
from miseval.core import evaluate


# Note:
# some dict which says if metric needs argmax before passing if probabilities==True
