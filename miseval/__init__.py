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
from miseval.dice import calc_DSC_Enhanced as calc_DSC
# Intersection-over-Union
from miseval.jaccard import *
from miseval.jaccard import calc_IoU_Sets as calc_IoU
# Sensitivity
from miseval.sensitivity import *
from miseval.sensitivity import calc_Sensitivity_Sets as calc_Sensitivity
# Specificity
from miseval.specificity import *
from miseval.specificity import calc_Specificity_Sets as calc_Specificity
# Precision
from miseval.precision import *
from miseval.precision import calc_Precision_Sets as calc_Precision
# Accuracy
from miseval.accuracy import *
from miseval.accuracy import calc_Accuracy_Sets as calc_Accuracy
# Area under the ROC
from miseval.auc import *
from miseval.auc import calc_AUC_trapezoid as calc_AUC
# Cohen's Kappa
from miseval.kappa import *
# Hausdorff Distance
from miseval.hausdorff import calc_SimpleHausdorffDistance, \
                              calc_AverageHausdorffDistance
# Volumetric Similarity
from miseval.volumetric_similarity import *
# Matthews Correlation Coefficient
from miseval.mcc import *
# Boundary Distances
from miseval.boundary_distance import *
# Hinge loss
from miseval.hinge import *
# Entropy-based metrics
from miseval.entropy import *

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
    "AdjustedRandIndex": calc_AdjustedRandIndex,
    "AUC": calc_AUC,
    "AUC_trapezoid": calc_AUC,
    "KAP": calc_Kappa,
    "Kappa": calc_Kappa,
    "CohensKappa": calc_Kappa,
    "Sensitivity": calc_Sensitivity,
    "SENS": calc_Sensitivity,
    "TPR": calc_Sensitivity,
    "TruePositiveRate": calc_Sensitivity,
    "Recall": calc_Sensitivity,
    "SPEC": calc_Specificity,
    "Specificity": calc_Specificity,
    "TNR": calc_Specificity,
    "TrueNegativeRate": calc_Specificity,
    "PREC": calc_Precision,
    "Precision": calc_Precision,
    "VS": calc_VolumetricSimilarity,
    "VolumetricSimilarity": calc_VolumetricSimilarity,
    "HD": calc_SimpleHausdorffDistance,
    "HausdorffDistance": calc_SimpleHausdorffDistance,
    "AHD": calc_AverageHausdorffDistance,
    "AverageHausdorffDistance": calc_AverageHausdorffDistance,
    "MatthewsCorrelationCoefficient": calc_MCC,
    "MCC": calc_MCC,
    "MCC_normalized": calc_MCC_Normalized,
    "nMCC": calc_MCC_Normalized,
    "MCC_absolute": calc_MCC_Absolute,
    "aMCC": calc_MCC_Absolute,
    "BoundaryDistance": calc_Boundary_Distance,
    "Distance": calc_Boundary_Distance,
    "BD": calc_Boundary_Distance,
    "Hinge": calc_Hinge,
    "HingeLoss": calc_Hinge,
    "CE": calc_CrossEntropy,
    "CrossEntropy": calc_CrossEntropy,
}

#-----------------------------------------------------#
#                     Core Imports                    #
#-----------------------------------------------------#
from miseval.core import evaluate
