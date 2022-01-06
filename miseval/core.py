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
from miseval import metric_dict

#-----------------------------------------------------#
#             Core Hub Function: Evaluate             #
#-----------------------------------------------------#
""" blabla. todo

Metric Parameter:
    The miseval eavluate function can be run with different metrics as backbone.
    You can pass the following options to the metric parameter:
    - String naming one of the metric labels, for example "ConfusionMatrix"
    - Directly passing a metric function, for example calc_ConfusionMatrix (from confusion_matrix.py)
    - Passing a custom metric function

    List of metrics : See miseval/__init__.py under section "Access Functions to Metric Functions"

Arguments:
    truth (NumPy Matrix):           A data generator which will be used for inference.
    pred (NumPy Matrix):             Instance of a AUCMEDI neural network class.
    metric (String or Function):                NumPy Array of classification prediction encoded as OHE (output of a AUCMEDI prediction).
    multi_class (Boolean):                    XAI method class instance or index. By default, GradCAM is used as XAI method.
    classes (Integer):                 Layer name of the convolutional layer for heatmap computation. If None, the last conv layer is used.
    probabilities (Boolean):                      Transparency value for heatmap overlap plotting on input image (range: [0-1]).
"""
def evaluate(truth, pred, metric, multi_class=False, classes=2,
             probabilities=False):
    # Initialize metric function
    if isinstance(metric, str) and metric in metric_dict:
        eval_metric = metric_dict[metric]
    else : eval_metric = metric

    # if multi_class == false and classes==2
    # run only on class==1

    # else iterate with for loop over it

   ...
   if multi_class : return results
   else : return results[1]
